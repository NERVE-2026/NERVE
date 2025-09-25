#%%
import os
import math
import argparse
import yaml
import time
import numpy as np
from datetime import datetime

from data_processor.dataset_utils import build_finetuning_dataset
from models.encoder import EEGModel
from utils import transfer_weights, cosine_scheduler, set_seed, freeze_weights_for_eval, setup_ddp, cleanup_ddp
from exp.exp_svreeg_finetune import Exp_SVREEG_Finetune

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description='Subject-variability-robust EEG encoder Finetuning')
parser.add_argument('--model', type=str, default='base')
parser.add_argument('--fm_configs', type=str, default='configs/eeg_fm/eegfm_base.yaml')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--eval', action='store_true', default=False)
# dataset & dataloader
parser.add_argument('--root_path', type=str, default='../dataset/processed/eeg')
parser.add_argument('--dset_name', type=str, default='HCI-Tagging_emotion')
parser.add_argument('--seq_len', type=int, default=25600)
parser.add_argument('--patch_size', type=int, default=200)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--stride', type=int, default=800)

# evaluation setting
parser.add_argument('--eval_type', type=str, default='finetune', choices=['finetune', 'linearprobe', 'partial'])
parser.add_argument('--n_finetune_layer', type=int, default=0)

# low-resource setting
parser.add_argument('--resource_ratio', type=float, default=1.0, help='Ratio of the training data to use for low-resource setting')

# define vq encoder & decoder
parser.add_argument('--out_channel', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=9)
parser.add_argument('--d_model', type=int, default=200)
parser.add_argument('--num_head', type=int, default=10)
parser.add_argument('--num_group', type=int, default=6)
parser.add_argument('--qkv_bias', type=bool, default=False)
parser.add_argument('--qk_norm', type=str, default=None)
parser.add_argument('--qk_scale', type=bool, default=None)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--attn_dropout', type=float, default=0.1)
parser.add_argument('--norm', type=str, default='layernorm')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--use_abs_pos_emb', type=bool, default=True)
parser.add_argument('--use_ch_emb', type=bool, default=True)
parser.add_argument('--head_type', type=str, default='onelayer', choices=['pretrain', 'onelayer', 'twolayer', 'norm-onelayer', 'norm-twolayer', 'allpatch-onelayer', 'allpatch-twolayer', 'conv-onelayer', 'conv-twolayer', 'conv-norm-onelayer', 'conv-norm-twolayer'])
parser.add_argument('--use_mean_pooling', type=bool, default=False)

# training
parser.add_argument('--train_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--warmup_lr_init', type=float, default=1e-6)
parser.add_argument('--warmup_epochs', type=float, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--clip_grad_norm', type=float, default=0)
parser.add_argument('--pretrain_path', type=str, default='outputs/pretrain_checkpoints')
parser.add_argument('--save_path', type=str, default='outputs/downstream_checkpoints')

# gpu
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')

# DDP 관련 추가
parser.add_argument('--use_ddp', action='store_true', default=False)
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23456')

# vq task version
parser.add_argument('--vq_task', type=int, default=2)
parser.add_argument('--enc_type', type=str, default='epa2')

# wandb
parser.add_argument('--use_wandb', action='store_true', default=False)
parser.add_argument('--project', type=str, default='class-finetune')
parser.add_argument('--run_name', type=str, default='HCI-Tagging_emotion')
parser.add_argument('--wandb_dir', type=str, default='./outputs/wandb')

parser.add_argument('--comment', type=str, default='')
parser.add_argument('--plus', type=str, default='')  # model 이름 뒤에 추가로 붙는 문자

#%%

def get_models(args):
    eeg_encoder = EEGModel(
        seq_len=args.seq_len, patch_size=args.patch_size, in_channel=1, out_channel=args.out_channel, num_class=args.num_class, num_group=args.num_group,
        d_model=args.d_model, num_head=args.num_head, num_layer=args.e_layers, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, qk_norm=args.qk_norm,
        dropout=args.dropout, attn_dropout=args.attn_dropout, norm=args.norm, activation=args.activation, 
        use_abs_pos_emb=args.use_abs_pos_emb, use_ch_emb=args.use_ch_emb, use_mean_pooling=args.use_mean_pooling, 
        encoder_type=args.enc_type, head_type=args.head_type, sample_duration=args.target_sample_duration, target_channel=args.target_num_channel
    )
    device = f'cuda:{args.rank}' if args.use_ddp else args.device
    eeg_encoder = eeg_encoder.to(device)

    eeg_encoder = transfer_weights(args.weights_path, eeg_encoder, device, key='encoder_state_dict', exclude_head=True)
    eeg_encoder = freeze_weights_for_eval(eeg_encoder, args.eval_type, n_fixed_layers=args.e_layers - args.n_finetune_layer)
    
    if args.use_ddp:
        eeg_encoder = DDP(eeg_encoder, device_ids=[args.rank], find_unused_parameters=True)

    n_learnable_params = sum(p.numel() for p in eeg_encoder.parameters() if p.requires_grad)
    n_fixed_params = sum(p.numel() for p in eeg_encoder.parameters() if not p.requires_grad)

    if not args.use_ddp or args.rank == 0:  # 0번 프로세스에서만 출력
        print(f'Number of learnable parameters: {n_learnable_params / 1e6} M')
        print(f'Number of fixed parameters: {n_fixed_params / 1e6} M')

    return eeg_encoder

#%%
def get_dataset(args):
    base_dir = '../dataset/processed/eeg/'
    
    """ 1. Load Dataset """
    finetuning_datasets_dict = {
            'HCI-Tagging_ERP': 32,     # new
            'SEED-V': 62,              # existing
            'DEAP': 32,                # new
            'HCI-Tagging_emotion': 32, # new
            'BCI-NER-Challenge': 56,   # new
            'SEED-VIG': 17,            # existing
        
            # === MS (7/7) ===
            'NMT(Events)': 19,         # new 
            'NMT(Scalp-EEG)': 19,      # new
            'TUSL': 20,                # existing  
            'TUAB': 21,                # existing
            'TUEV': 23,                # existing
            'CHB-MIT': 23,             # existing, but different window
            
            # === YJ (4/4) ===
            'Sleep_EDF': 2,            # new
            'Physionet_MI_EEG': 64,    # existing
            'High_Gamma': 133,         # new
            'ISRUC_Sleep': 6,          # existing, but different window
        }
    
    fine_splitted_datasets = [
        # === SY ===
        'BCI-NER-Challenge',
        # === MS ===
        'NMT(Scalp-EEG)',
        'TUAB',
        'TUSZ',   
        'TUEV',   
        ]

    finetune_dataset_time_window_dict = {
        'TUEV' : 5,
        'TUAB' : 10,
        'TUSL': 5,
        'Physionet_MI_EEG': 4,
        'SEED-VIG': 8,
        'SEED-V': 1,
        'ISRUC_Sleep': 5, # new window, original: 30
        'CHB-MIT': 10, # new window, original 10
    }
    finetuning_dataset = args.dset_name
    finetuning_dataset_time_window = finetune_dataset_time_window_dict.get(finetuning_dataset, 10)  # default 10 seconds if not specified
    
    data_load_start = time.time()
    if finetuning_dataset in fine_splitted_datasets:
        # (2) official splitted datasets
        dataset_train, ch_idxes = build_finetuning_dataset(
            base_dir=base_dir,
            dataset=finetuning_dataset, 
            time_window=finetuning_dataset_time_window, 
            stride_size=200*finetuning_dataset_time_window, 
            start_percentage=0, end_percentage=1,
            sfreq=200, dataset_split='train', rank=args.rank
        )
        
        if 'validation_npy_files' in os.listdir(os.path.join(base_dir, finetuning_dataset)):
            dataset_val, _ = build_finetuning_dataset(
                base_dir=base_dir,
                dataset=finetuning_dataset, 
                time_window=finetuning_dataset_time_window, 
                stride_size=200*finetuning_dataset_time_window, 
                start_percentage=0, end_percentage=1,
                sfreq=200, dataset_split='validation', rank=args.rank
            )
        else:
            from torch.utils.data import random_split
            split = [0.8, 0.2]
            total_len = len(dataset_train)
            train_len = int(len(dataset_train) * split[0])
            val_len = total_len - train_len
            dataset_train, dataset_val = random_split(dataset_train, [train_len, val_len])

        dataset_test, _ = build_finetuning_dataset(
            base_dir=base_dir,
            dataset=finetuning_dataset, 
            time_window=finetuning_dataset_time_window, 
            stride_size=200*finetuning_dataset_time_window, 
            start_percentage=0, end_percentage=1,
            sfreq=200, dataset_split='test', rank=args.rank,
        )
    else:
        dataset_train, ch_idxes = build_finetuning_dataset(
            base_dir=base_dir,
            dataset=finetuning_dataset, 
            time_window=finetuning_dataset_time_window, 
            stride_size=200*finetuning_dataset_time_window, 
            start_percentage=0, end_percentage=1,
            sfreq=200, dataset_split='train', rank=args.rank,
        )
        dataset_val, _ = build_finetuning_dataset(
                base_dir=base_dir,
                dataset=finetuning_dataset, 
                time_window=finetuning_dataset_time_window, 
                stride_size=200*finetuning_dataset_time_window, 
                start_percentage=0, end_percentage=1,
                sfreq=200, dataset_split='validation', rank=args.rank,
            )
        dataset_test, _ = build_finetuning_dataset(
            base_dir=base_dir,
            dataset=finetuning_dataset, 
            time_window=finetuning_dataset_time_window, 
            stride_size=200*finetuning_dataset_time_window, 
            start_percentage=0, end_percentage=1,
            sfreq=200, dataset_split='test', rank=args.rank,
            )
        
    if args.resource_ratio < 1.0:
        # Low-resource setting: subsample the training dataset
        num_samples = int(len(dataset_train) * args.resource_ratio)
        indices = torch.randperm(len(dataset_train))[:num_samples]
        dataset_train = torch.utils.data.Subset(dataset_train, indices)
            
    if args.rank==0:  # 0번 프로세스에서만 출력
        print(f"Data loading time: {(time.time() - data_load_start)//60} minutes")

    if args.use_ddp:
        sampler_train = DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context=mp.get_context('fork') if args.use_ddp else None
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args.batch_size*1.5),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context=mp.get_context('fork') if args.use_ddp else None
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=int(args.batch_size*1.5),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context=mp.get_context('fork') if args.use_ddp else None
    )
    
    if isinstance(dataset_train, torch.utils.data.Subset):
        labels = dataset_train.dataset.get_labels()
    else:
        labels = dataset_train.get_labels()  
    num_classes = len(torch.unique(torch.tensor(labels)))

    if args.rank==0:
        print(f"Dataset: {finetuning_dataset} | num_channels: {len(ch_idxes)}, num_classes: {num_classes} | "
            f"Train: {len(data_loader_train)} batches, Val: {len(data_loader_val)} batches, Test: {len(data_loader_test)} batches")

    return data_loader_train, data_loader_val, data_loader_test, ch_idxes, num_classes, finetuning_dataset_time_window


def main(rank, world_size, args):

    if args.use_ddp:
        setup_ddp(rank, world_size, args.dist_url)
        args.rank = rank
        args.world_size = world_size
        args.device = torch.device(f'cuda:{rank}')

    train_loader, val_loader, test_loader, ch_idxes, num_class, sample_duration = get_dataset(args)
    data_loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    args.num_class = num_class
    args.target_sample_duration = sample_duration
    args.target_num_channel = len(ch_idxes)

    eeg_encoder = get_models(args)
    n_iters_per_epoch = len(train_loader)
    lr_schedule_values = cosine_scheduler(
                                        args.lr, args.min_lr, args.train_epochs, n_iters_per_epoch, 
                                        warmup_epochs=args.warmup_epochs, warmup_lr_init=args.warmup_lr_init
                                        )

    if args.rank==0:
        print(args)

    exp = Exp_SVREEG_Finetune(args, eeg_encoder, data_loader, ch_idxes)
    if args.eval:
        exp.evaluate()
    else:
        exp.train(lr_schedule_values)
        exp.evaluate()

def replace_fm_arguments(args, fm_args):
    names = [
        'seq_len', 'patch_size', 'in_channel', 'out_channel', 'num_class', 'num_group', 'd_model', 'num_head',
        'e_layers', 'qkv_bias', 'qk_scale', 'qk_norm', 'dropout', 'attn_dropout', 'norm', 'activation',
        'use_abs_pos_emb', 'use_ch_emb', 'use_mean_pooling', 'encoder_type'
    ]
    for key, val in fm_args.items():
        if key in names:
            setattr(args, key, val)
    return args


if __name__ == '__main__':

    args = parser.parse_args()
    with open(args.fm_configs, 'r') as f:
        fm_args = yaml.safe_load(f)
    args = replace_fm_arguments(args, fm_args)
    set_seed(args.seed)
    args.num_patch = args.seq_len // args.patch_size
    
    args.weights_path = os.path.join(args.pretrain_path, 'eegmodel', args.model, 'encoder_ckpt_best.pth')
    args.task_path = os.path.join(args.save_path, args.model, args.dset_name, str(args.seed), args.comment)
    args.res_path = os.path.join('results', args.model+args.plus, args.dset_name)
    
    if not os.path.exists(args.task_path):
        os.makedirs(args.task_path)
        
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    if args.use_ddp:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.world_size = len(device_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

        # If launched via torchrun / torch.distributed.launch, use env vars (recommended)
        local_rank_env = os.environ.get('LOCAL_RANK') or os.environ.get('SLURM_LOCALID') or os.environ.get('OMPI_COMM_WORLD_RANK')
        if local_rank_env is not None:
            # torchrun launched processes: call main directly per-process
            local_rank = int(local_rank_env)
            args.rank = local_rank
            args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
            # device assignment
            args.device = torch.device(f'cuda:{args.rank}')
            main(args.rank, args.world_size, args)
        else:
            # Fallback: spawn processes locally, force 'fork' start method to inherit open file handles
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass
            mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
    
    else:

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            
        if not args.use_gpu:
            args.device = torch.device('cpu')
        else:
            if args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
                print('Use Multi GPU: cuda:{}'.format(args.devices))
                
            args.device = torch.device('cuda:{}'.format(args.gpu))
            if not args.use_multi_gpu:
                print('Use GPU: cuda:{}'.format(args.gpu))

        main(0, 1, args)  # rank=0, world_size=1로 단일 프로세스 실행