#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_squared_error
import torch.distributed as dist


#%%
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # 아래 옵션은 재현성을 높이지만, 속도가 느려질 수 있음
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def setup_ddp(rank, world_size, dist_url):
    """DDP 환경 초기화"""
    os.environ['MASTER_ADDR'] = dist_url.split('://')[1].split(':')[0]
    os.environ['MASTER_PORT'] = dist_url.split(':')[-1]
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """DDP 정리"""
    dist.destroy_process_group()

def freeze_weights_for_eval(model, eval_type, n_fixed_layers):
    if eval_type=='linearprobe':
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    elif eval_type == 'partial':
        for name, param in model.named_parameters():
            # 트랜스포머 레이어 이름이 attention.attn_layers.{i}. 으로 시작함
            if name.startswith('attention.attn_layers.'):
                layer_num = int(name.split('.')[2])
                if layer_num < n_fixed_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False               

    return model


class Evaluator:
    def __init__(self):
        pass
    
    def get_multiclass_metrics(self, targets, preds):
        acc = balanced_accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        kappa = cohen_kappa_score(targets, preds)
        cm = confusion_matrix(targets, preds)
        return acc, f1, kappa, cm

    def get_regression_metrics(self, targets, preds):
        corr = np.corrcoef(targets, preds)[0, 1]
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        return corr, r2, rmse
    
    def get_binaryclass_metrics(self, targets, preds, scores):
        acc = balanced_accuracy_score(targets, preds)
        auprc = average_precision_score(targets, scores)
        auc = roc_auc_score(targets, scores)
        prec = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        cm = confusion_matrix(targets, preds)
        return acc, auprc, auc, prec, recall, cm

def ddp_reduce(metrics, world_size, device):
    metrics = torch.tensor(metrics, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if world_size != 1:
        metrics /= world_size
    return metrics.cpu().numpy()

def cosine_scheduler(
    base_lr,          # 초기 학습률 (cosine 스케줄의 최대값)
    min_lr,           # 최저 학습률 (cosine 스케줄의 최소값)
    epochs,           # 총 학습 에폭 수
    steps_per_epoch,  # 한 에폭당 스텝(배치) 수
    warmup_epochs=0,  # warmup을 몇 에폭 할지
    warmup_lr_init=0  # warmup 시작 시점의 LR (대개 0 또는 매우 작은 값)
):
    """
    매 스텝마다 LR을 미리 계산한 배열을 반환.
    1) warmup 구간: 선형 증가 (warmup_epochs 동안)
    2) 이후: Cosine decay (base_lr -> min_lr)
    
    returns:
      lr_schedule: shape (total_steps,) 
                   각 스텝(0 ~ total_steps-1)에 대응하는 learning rate
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_schedule = np.zeros(total_steps, dtype=np.float32)

    # 1) Warmup 구간 (선형 증가)
    if warmup_steps > 0:
        lr_schedule[:warmup_steps] = np.linspace(warmup_lr_init, base_lr, warmup_steps)

    # 2) Cosine 구간
    # warmup_steps ~ total_steps-1
    steps = np.arange(total_steps - warmup_steps)
    lr_schedule[warmup_steps:] = np.array([min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / len(steps))) for t in steps])

    return lr_schedule

def transfer_weights(weights_path, model, device, key, exclude_head=True):
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)[key]
    matched_layers = 0
    unmatched_layers = []

    for name, param in model.state_dict().items():
        if exclude_head and 'head' in name: continue
        if name in state_dict:
            matched_layers += 1
            input_param = state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
        
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


class Augmentation(nn.Module):
    def __init__(self, noise_sample_ratio=0.5, noise_std=0.05):

        super().__init__()
        self.noise_sample_ratio = noise_sample_ratio
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_aug = x.clone()
        if self.noise_std > 0:
            return self.add_noise(x_aug)
        else:
            return x_aug

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        B, C, L = x.shape
        noise = torch.randn((B, C, L), device=x.device) * self.noise_std
        mask = torch.rand((B, ), device=x.device) < self.noise_sample_ratio
        x[mask] += noise[mask]
        # x[mask] *= 0
        return x

def create_mask(x, mask_prob=0.5):
    B, C, N, T = x.shape
    mask = torch.rand((B, C, N), device=x.device) < mask_prob
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask.bool()

def l2norm(z):
    return F.normalize(z, p=2, dim=-1)

def KoLeo_loss(emb):
    # emb: (bs, num_channel, num_patch, d_model)
    emb = emb.mean(dim=2).mean(dim=1) # (bs, d_model)
    emb = l2norm(emb)
    dist_mat = torch.cdist(emb, emb, p=2) ** 2 # (bs, bs)
    min_indices = torch.topk(dist_mat, 2, largest=False).indices[:, 1]
    loss = - torch.log(dist_mat.gather(1, min_indices.unsqueeze(1))).mean()
    return loss

def get_activation_fn(activation):
    if activation.lower()=='relu':
        return nn.ReLU
    elif activation.lower()=='gelu':
        return nn.GELU
    
def get_normalization_fn(norm):
    if 'layer' in norm.lower():
        return LayerNorm
    elif 'batch' in norm.lower():
        return BatchNorm
    elif 'rms' in norm.lower():
        return nn.RMSNorm

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln(x)

class BatchNorm(nn.Module):
    def __init__(self, d_model):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(d_model)
    
    def forward(self, x):
        return self.bn(x.transpose(-2, -1)).transpose(-2, -1)

# %%
