#%%
import os
import math
import argparse
import numpy as np
import time
from datetime import datetime

from models.quantizer import VectorQuantizeEncoder, VectorQuantizeDecoder, VectorQuantizeBackbonePretrain
from data_processor.dataset_utils import build_pretraining_dataset
from utils import Augmentation, cosine_scheduler, transfer_weights, set_seed, setup_ddp, cleanup_ddp
from exp.exp_nrvq_pretrain import Exp_NRVQ
from configs.arguments import _load_yaml, build_vq_pretrain_parser

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

#%%
parser = argparse.ArgumentParser(description='Noise-Robust Vector Quantizer Pre-training')
parser.add_argument('--vq_configs', type=str, default='configs/vector_quantizer/vq.yaml')

#%%

def get_models(args):

    vq_encoder = VectorQuantizeEncoder(
        seq_len=args.seq_len, patch_size=args.patch_size, in_channel=1, out_channel=args.out_channel, num_class=0, num_group=args.num_group,
        num_tokens=args.num_tokens, d_codebook=args.d_codebook, d_model=args.d_model, num_head=args.num_head, num_layer=args.e_layers, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, qk_norm=args.qk_norm,
        dropout=args.dropout, attn_dropout=args.attn_dropout, norm=args.norm, activation=args.activation, use_abs_pos_emb=args.use_abs_pos_emb, use_ch_emb=args.use_ch_emb, use_mean_pooling=args.use_mean_pooling,
        decay=args.ema_decay, eps=args.eps, statistic_code_usage=args.statistic_code_usage, kmeans_init=args.kmeans_init, codebook_init_path=args.codebook_init_path, 
        encoder_type=args.enc_type
    )
    vq_decoder = VectorQuantizeDecoder(
        seq_len=args.num_patch, patch_size=1, in_channel=args.d_codebook, out_channel=args.d_model, num_class=0, d_out=args.patch_size, num_group=args.num_group, 
        d_model=args.d_model, num_head=args.num_head, num_layer=args.d_layers, qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, qk_norm=args.qk_norm,
        dropout=args.dropout, attn_dropout=args.attn_dropout, norm=args.norm, activation=args.activation, use_abs_pos_emb=args.use_abs_pos_emb, use_ch_emb=args.use_ch_emb, use_mean_pooling=args.use_mean_pooling,
        vq_task=args.vq_task, encoder_type=args.enc_type
        )
    
    augmentation = get_augmentation(args)    
    vq_backbone = VectorQuantizeBackbonePretrain(args, vq_encoder, vq_decoder, augmentation)
    
    if args.use_ddp:
        vq_backbone = vq_backbone.to(args.rank)
        vq_backbone = DDP(vq_backbone, device_ids=[args.rank], find_unused_parameters=True)
    else:
        vq_backbone = vq_backbone.to(args.device)
        if args.use_multi_gpu & (torch.cuda.device_count() > 1):
            vq_backbone = nn.DataParallel(vq_backbone, device_ids=args.device_ids)
    
    n_learnable_params = sum(p.numel() for p in vq_backbone.parameters() if p.requires_grad)
    n_fixed_params = sum(p.numel() for p in vq_backbone.parameters() if not p.requires_grad)
    
    if args.rank == 0:  # 0번 프로세스에서만 출력
        print(f'Number of learnable parameters: {n_learnable_params / 1e6} M')
        print(f'Number of fixed parameters: {n_fixed_params / 1e6} M')
    
    return vq_backbone


def get_augmentation(args):
    augmentation = Augmentation(noise_sample_ratio=args.noise_sample_ratio, noise_std=args.noise_std)    
    return augmentation

def get_dataset(args):
    base_dir = args.root_path
    
    """ 1. Load Dataset """
    # prepare pretrain_datasets, pretrain_datasets_time_window  
    pretrain_datasets_dict = {
            # === SY (9/9(P300 길이가 짧아 제외)) ===
            'Mumtaz2016': 19,
            'SEED-IV': 62,
            'BCIC2020-3': 64,
            'DREAMER': 14,
            'SHU-MI': 32, 
            'Berlin_dsr': 26 , 
            'Berlin_nback': 26,
            'Berlin_wg': 26,   
            # #  === MS (4/6) ===
            'Neonate': 19,
            'Fatigueset': 4,
            'Mental Arithmetic': 19,
            'Stew': 14,
            'TUSZ': 20,
            # # === YJ (11/12) ===        
            'BCI_IV_1': 59,   # o
            'SPIS_Resting_State_Dataset': 64,
            'Grasp_and_Lift_EEG_Challenge': 32,
            'Inria_BCI_Challenge': 56,
            'Target_Versus_Non_Target': 32,
            'Raw_EEG_Data': 64,
            'BCI_IV_2a': 22,   
            'BCI_IV_2b': 3,    
            'Emobrain': 64,    
            'MoBI': 60,        
            'Resting_State_EEG_data': 64,
            }

    pretrain_datasets_dict.update({f'Siena_{i}': 28 for i in range(11)})
    pretrain_datasets_dict.update({f'2018_PhysioNet_Challenge_{i}': 6 for i in range(5)})
    # pretrain_datasets_dict.update({f'TUEG_{i}': 19 for i in range(24)})  # TUEG_0 ~ TUEG_23

    if args.rank == 0:
        print(pretrain_datasets_dict)
    pretrain_datasets = list(pretrain_datasets_dict.keys())
    # pretrain_datasets_time_window = np.full((len(pretrain_datasets),), 30)  # 30초로 설정, 필요시 조정 가능
    pretrain_datasets_time_window = [256//n_channel for n_channel in pretrain_datasets_dict.values()]

    data_load_start = time.time()
    dataset_train_list, train_ch_idx_list = build_pretraining_dataset(
        base_dir=base_dir,
        datasets=[[dataset] for dataset in pretrain_datasets], 
        time_window=pretrain_datasets_time_window, 
        stride_size=args.stride, 
        start_percentage=0, end_percentage=1,
        sfreq=200, rank=args.rank,
        )
    total_timestamps = sum(dataset.get_total_timestamps() for dataset in dataset_train_list)

    if args.rank==0:  # 0번 프로세스에서만 출력
        print(f"Total timestamps in pretraining datasets: {total_timestamps}")
        print(f'Total hours in pretraining datasets: {total_timestamps / (200 * 3600):.3f} hours')
        print(f"Data loading time: {(time.time() - data_load_start)//60} minutes")

    # # cls 토큰 빼기
    # train_ch_idx_list = [list(np.array(ch_idx[1:]) - 1) for ch_idx in train_ch_idx_list]

    sampler_train_list = []
    for dataset in dataset_train_list:
        if args.use_ddp:
            sampler_train = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset)
        sampler_train_list.append(sampler_train)
    
    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # num_workers=0,
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'), 
        )
        data_loader_train_list.append(data_loader_train)

    return data_loader_train_list, train_ch_idx_list

def main(rank, world_size, args):

    if args.use_ddp:
        setup_ddp(rank, world_size, args.dist_url)
        args.rank = rank
        args.world_size = world_size
        args.device = torch.device(f'cuda:{rank}')
        
    data_loader_train_list, train_ch_idx_list = get_dataset(args)
    n_iters_per_epoch = sum([len(data_loader) for data_loader in data_loader_train_list])
    lr_schedule_values = cosine_scheduler(
                                        args.lr, args.min_lr, args.train_epochs, n_iters_per_epoch, 
                                        warmup_epochs=args.warmup_epochs, warmup_lr_init=args.warmup_lr_init
                                        )
                                        
    if rank == 0:
        print('Scheduler parameters:')
        print(f'  Initial learning rate: {args.lr}')
        print(f'  Minimum learning rate: {args.min_lr}')
        print(f'  Warmup learning rate: {args.warmup_lr_init}')
        print(f'  Warmup epochs: {args.warmup_epochs}')
        print(f'  Total epochs: {args.train_epochs}')
        print(f'  Total iterations per epoch: {n_iters_per_epoch}')
    
    vq_model = get_models(args)
    exp = Exp_NRVQ(args, vq_model, data_loader_train_list, train_ch_idx_list)
    exp.pretrain(lr_schedule_values)
    # exp.evaluate_vq_learning()
    
    if args.use_ddp:
        cleanup_ddp()

if __name__ == '__main__':

    known, _ = parser.parse_known_args()
    cfgs = _load_yaml(known.vq_configs)
    parser = build_vq_pretrain_parser(defaults=cfgs)
    args = parser.parse_args()

    set_seed(args.seed)
    args.num_patch = args.seq_len // args.patch_size
    args.save_path = os.path.join(args.pretrain_path, f'quantizer')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
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
        

        # # DDP 멀티프로세싱 시작
        # mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
        
    else:
        # 기존 DataParallel 방식
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = list(range(len(device_ids)))
            
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
# %%
