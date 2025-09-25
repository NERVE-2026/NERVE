#%%



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat
from models.encoder import EEGModel
from utils import get_activation_fn, get_normalization_fn

#%%
def std_norm(x):
    mean = torch.mean(x, dim=(1,2,3), keepdim=True)
    std = torch.std(x, dim=(1,2,3), keepdim=True)
    return (x - mean) / std

def sum_norm(x):
    sum_x = torch.sum(x, dim=(1,2,3), keepdim=True)
    return x / sum_x.clamp(min=1e-8)  # Avoid division by zero    

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

#%%

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters) # randomly permuted token embeddings

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 
        if codebook_init_path == '':   
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                # kmeans_init=True -> 여기 들어감
                weight = torch.zeros(num_tokens, codebook_dim) 
            self.register_buffer('initted', torch.Tensor([not kmeans_init])) # self.initted = False
        else:
            # 초기 코드북 벡터 불러옴
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True])) # self.initted = True
            
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):

        # 이거 안지나감
        if self.initted:
            # 초기 코드북 벡터 저장된 것이 있으면 여기로 들어와서 그냥 return
            # VQ 사전학습 후 finetuning 시에는 여기로 들어옴
            return
        
        # print("Performing Kmeans init for codebook")
        
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True) # perform k-means clustering
        self.weight.data.copy_(embed) # 
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight) # self.weight에서 embed_id에 해당하는 row를 가져옴

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)   


class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        
        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path) # 초기 코드북 벡터
        
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens)) # num codebook
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()
        
        # track codebook utilization
        self.register_buffer('epoch_codebook_usage', torch.zeros(self.num_tokens, dtype=torch.bool)) # num codebook
        self.register_buffer('epoch_codebook_count', torch.zeros(self.num_tokens, dtype=torch.int)) # num codebook
            
    # Updates the epoch-level codebook usage based on the current batch.
    def update_codebook_usage_epoch(self, encodings):
        with torch.no_grad():
            used_codebooks = encodings.int().sum(dim=0)
            self.epoch_codebook_count += used_codebooks
            self.epoch_codebook_usage |= used_codebooks>0
            
    def get_unused_codebook_count_epoch(self):
        with torch.no_grad():
            unused_count = (~self.epoch_codebook_usage).sum().item()
        return unused_count
    
    def get_codebook_usage_entropy_epoch(self):
        with torch.no_grad():
            # epoch_codebook_usage: [num_tokens] (bool)
            # 실제 사용된 횟수 카운트 (encodings의 합)
            usage_count = self.epoch_codebook_count.float()
            total_used = usage_count.sum().item()
            if total_used == 0:
                return 0.0
            # 확률 분포로 변환
            prob = usage_count / total_used
            # 엔트로피 계산 (log2 기준)
            entropy = -(prob[prob > 0] * prob[prob > 0].log2()).sum().item()
        return entropy

    def reset_epoch_codebook_usage(self):
        self.epoch_codebook_usage.zero_()
        self.epoch_codebook_count.zero_()
    
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        # b, d_model, c (h), num_patch (w)
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'  'b, channel, num_patch, patch_size (200)
        z = rearrange(z, 'b c h w -> b h w c') # b, num_patch, patch_size, d_model
        z = l2norm(z) 
        z_flattened = z.reshape(-1, self.codebook_dim).contiguous() # 
        self.embedding.init_embed_(z_flattened)
        
        # distance between patch embeddings and codebook vectors
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n' 
        
        # codebook indices
        encoding_indices = torch.argmin(d, dim=1)

        # codebook vectors
        z_q = self.embedding(encoding_indices).view(z.shape).contiguous()
        
        # codebook indices to one-hot vector
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     

        # EMA update of codebook usage statistics
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        if self.training and self.embedding.update:
            #EMA cluster size
            bins = encodings.sum(0) # count for each codebook vector (각 클러스터, 코드북 별 할당된 패치수)
            self.all_reduce_fn(bins) 

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
                        
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)

            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)
            
        # update codebook usage
        self.update_codebook_usage_epoch(encodings)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous() # (bs, c, num_patch, d_model) -> (bs, d_model, c, num_patch)
        
        
        return z_q, loss, encoding_indices


#%%

class VectorQuantizeEncoder(nn.Module):
    def __init__(self, seq_len, patch_size, in_channel, out_channel, num_class, num_group, num_tokens, d_codebook, 
                 d_model, num_head, num_layer, qkv_bias=False, qk_norm=None, qk_scale=None, dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0,
                 norm='layernorm', activation='gelu', use_abs_pos_emb=True, use_ch_emb=True, use_mean_pooling=False,
                 decay=0.99, eps=1e-5, statistic_code_usage=True, kmeans_init=True, codebook_init_path='', encoder_type='epa'):
        
        super().__init__()
        if norm is not None:
            norm = get_normalization_fn(norm)
        if activation is not None:
            activation = get_activation_fn(activation)
        if qk_norm is not None:
            qk_norm = get_normalization_fn(qk_norm)
            
        self.encoder = EEGModel(seq_len, patch_size, in_channel, out_channel, num_class, num_group, 
                                d_model, num_head, num_layer, qkv_bias, qk_norm, qk_scale, dropout, attn_dropout, drop_path_rate,
                                norm, activation, use_abs_pos_emb, use_ch_emb, use_mean_pooling, encoder_type=encoder_type, head_type='identity')
        self.quantizer = NormEMAVectorQuantizer(num_tokens, d_codebook, 1, decay, eps, statistic_code_usage, kmeans_init, codebook_init_path)

        self.patch_size = patch_size
        self.d_model = d_model
        self.enc_to_vq_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.Tanh(), nn.Linear(self.d_model, d_codebook)
        )

    
    def encode(self, x, input_chans=None):
        batch_size, c, num_patch, patch_size = x.shape
        encoder_features = self.encoder(x, input_chans)
        """
        torch.cuda.amp.autocast(enabled=False) 코드는 PyTorch에서 제공하는 자동 혼합 정밀도(AMP) 기능의 사용 여부를 설정하는 데 사용됩니다. 
        AMP는 신경망의 계산 속도를 높이면서도 메모리 사용량을 줄이기 위해, 데이터의 정밀도를 자동으로 조절합니다.
        이 코드에서 enabled=False는 AMP 기능을 비활성화한다는 의미입니다. 
        즉, 이 코드 블록 내의 연산들은 모두 표준 정밀도(예: float32)를 사용하여 수행됩니다. 
        AMP를 사용하지 않으면, 계산 속도가 다소 느려질 수 있지만, 때로는 더 정확하거나 예측 가능한 결과를 얻기 위해 필요할 수 있습니다.
        """
        with torch.amp.autocast('cuda'):
            to_quantizer_features = self.enc_to_vq_projection(encoder_features.type_as(self.enc_to_vq_projection[-1].weight))

        to_quantizer_features = rearrange(to_quantizer_features, 'b c n d -> b d c n') # (bs, d_model, c, num_patches)
        quantize, loss, embed_ind = self.quantizer(to_quantizer_features)

        return quantize, embed_ind, loss

    def get_tokens(self, x, input_chans=None):
        quantize, embed_ind, loss = self.encode(x, input_chans)
        output = {}
        output['token'] = embed_ind
        output['quantize'] = quantize
        return output

    def get_codebook_indices(self, x, input_chans=None):
        return self.get_tokens(x, input_chans)['token']

    def forward(self, x, input_chans=None):
        batch_size, c, num_patch, patch_size = x.shape
        # x = rearrange(x, 'B C (N T) -> B C N T', T=self.patch_size)
        quantize, embed_ind, emb_loss = self.encode(x, input_chans)

        return quantize, emb_loss

#%%

"""
vq_task==0: labram vq task - amplitude prediction, phase prediction,
vq_task==1: neurolm vq task - temporal prediction, amplitude prediction,
vq_task==2: our vq task - temporal prediction, normalized-amplitude fft prediction
vq_task==3: our vq task - temporal prediction, original fft prediction
"""

class VectorQuantizeDecoder(nn.Module):
    def __init__(self, seq_len, patch_size, in_channel, out_channel, num_class, num_group, 
                 d_model, num_head, num_layer, d_out, qkv_bias=False, qk_norm=None, qk_scale=None, dropout=0.0, attn_dropout=0.0, drop_path_rate=0.0,
                 norm='layernorm', activation='gelu', use_abs_pos_emb=True, use_ch_emb=False, use_mean_pooling=False, vq_task=0, encoder_type='epa'):
        super().__init__()

        self.vq_task = vq_task
        if norm is not None:
            norm = get_normalization_fn(norm)
        if activation is not None:
            activation = get_activation_fn(activation)
        if qk_norm is not None:
            qk_norm = get_normalization_fn(qk_norm)

        # for decoder, seq_len=num_patch, patch_size=1, out_channel is not use.
        # d_out is the original patch sizes
        self.decoder = EEGModel(seq_len, patch_size, in_channel, out_channel, num_class, num_group,
                                d_model, num_head, num_layer, qkv_bias, qk_norm, qk_scale, dropout, attn_dropout, drop_path_rate,
                                norm, activation, use_abs_pos_emb, use_ch_emb, use_mean_pooling, encoder_type=encoder_type)
                                
        self.temporal_prediction = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_out)
        )

        self.amplitude_prediction = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_out)
        )

        self.phase_prediction = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, d_out)
        )

    def forward(self, quantize, input_chans=None):
        # quantize:  
        decoder_features = self.decoder(quantize, input_chans)
        if self.vq_task==0:
            pred_amp = self.amplitude_prediction(decoder_features)
            pred_phase = self.phase_prediction(decoder_features)
            return pred_amp, pred_phase
        elif self.vq_task==1:
            rec_temporal = self.temporal_prediction(decoder_features)
            pred_amp = self.amplitude_prediction(decoder_features)
            return rec_temporal, pred_amp
        else:
            rec_temporal = self.temporal_prediction(decoder_features)
            return rec_temporal

#%%

class VectorQuantizeBackbonePretrain(nn.Module):
    def __init__(self, args, encoder, decoder, augmentation):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.eeg_aug = augmentation
        
        if self.args.loss.lower() in 'mse_loss': self.loss_fn = F.mse_loss 
        elif self.args.loss.lower() in 'l1_loss': self.loss_fn = F.smooth_l1_loss
        elif self.args.loss.lower() in 'huber_loss': self.loss_fn = F.huber_loss
        else: raise ValueError('Invalid loss function')
        
        self.freq_loss_fn = F.l1_loss 
    
    def forward(self, x, input_chans=None):
        # fredf 처럼 x_rec의 fft와 x의 fft를 비교하는 방식으로 변경. 단일 head (temporal reconstruction head)만 필요
        
        x_noise = self.eeg_aug(x)
        x = rearrange(x, 'B C (N T) -> B C N T', T=self.encoder.patch_size)
        x_noise = rearrange(x_noise, 'B C (N T) -> B C N T', T=self.encoder.patch_size)
        quantize, emb_loss = self.encoder(x_noise, input_chans)
        if self.args.vq_task==0:
            x_fft = torch.fft.fft(x, dim=-1)
            amp = std_norm(torch.abs(x_fft))
            phase = std_norm(torch.angle(x_fft))
            rec_amp, rec_phase = self.decoder(quantize, input_chans)
            rec_amp_loss = self.calculate_temp_loss(rec_amp, amp)
            rec_phase_loss = self.calculate_temp_loss(rec_phase, phase)
            return emb_loss, rec_amp_loss, rec_phase_loss

        elif self.args.vq_task==1:
            x_fft = torch.fft.fft(x, dim=-1)
            amp = std_norm(torch.abs(x_fft))
            rec_temp, rec_amp = self.decoder(quantize, input_chans)
            rec_temp_loss = self.calculate_temp_loss(rec_temp, x)
            rec_amp_loss = self.calculate_temp_loss(rec_amp, amp)
            return emb_loss, rec_temp_loss, rec_amp_loss
        
        else:
            rec_temp = self.decoder(quantize, input_chans)
            rec_temp_loss = self.calculate_temp_loss(rec_temp, x)
            rec_freq_loss = self.calculate_freq_loss(rec_temp, x)
            return emb_loss, rec_temp_loss, rec_freq_loss
    
    def calculate_temp_loss(self, rec, target):
        rec = rearrange(rec, 'b n a c -> b (n a) c')
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def calculate_freq_loss(self, rec, target):
        if self.args.vq_task==2:
            # version 1
            x_fft = torch.fft.rfft(target, dim=-1)
            x_amp = sum_norm(torch.abs(x_fft))
            x_phase = torch.angle(x_fft)
            x_fft_norm = x_amp * torch.exp(1j * x_phase)
            
            x_rec_fft = torch.fft.rfft(rec.float(), dim=-1)
            x_rec_amp = sum_norm(torch.abs(x_rec_fft))
            x_rec_phase = torch.angle(x_rec_fft)
            x_rec_fft_norm = x_rec_amp * torch.exp(1j * x_rec_phase)
            
            rec_loss_freq = (x_fft_norm - x_rec_fft_norm).abs().mean()        
            return rec_loss_freq
        else:        
            # version 2
            x_fft = torch.fft.rfft(target, dim=-1)
            x_rec_fft = torch.fft.rfft(rec.float(), dim=-1)
            rec_loss_freq = (x_fft - x_rec_fft).abs().mean() # L1 loss on the complex frequency domain
            return rec_loss_freq