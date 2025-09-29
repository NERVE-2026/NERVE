#%%
import os
import math
import argparse
import numpy as np
import time
import yaml
from datetime import datetime

from data_processor.dataset_utils import build_pretraining_dataset
from models.quantizer import VectorQuantizeEncoder
from models.encoder import EEGModel, EncoderSVREEGPretrain
from utils import transfer_weights, cosine_scheduler, set_seed, setup_ddp, cleanup_ddp
from exp.exp_svreeg_pretrain import Exp_SVREEG_Pretrain
from configs.arguments import _load_yaml, build_fm_pretrain_parser

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

parser = argparse.ArgumentParser(description='Subject-variability-robust EEG encoder Pre-training')
parser.add_argument('--model', type=str, default='base')
parser.add_argument('--vq_configs', type=str, default='configs/vector_quantizer/vq.yaml')
parser.add_argument('--fm_configs', type=str, default='configs/eeg_fm/eegfm_base.yaml')

#%%
    
def get_models(vq_args, fm_args):
    eeg_encoder = EEGModel(
        seq_len=fm_args.seq_len, patch_size=fm_args.patch_size, in_channel=1, out_channel=fm_args.out_channel, num_class=fm_args.num_tokens, num_group=fm_args.num_group,
        d_model=fm_args.d_model, num_head=fm_args.num_head, num_layer=fm_args.e_layers, qkv_bias=fm_args.qkv_bias, qk_scale=fm_args.qk_scale, qk_norm=fm_args.qk_norm,
        dropout=fm_args.dropout, attn_dropout=fm_args.attn_dropout, norm=fm_args.norm, activation=fm_args.activation,
        use_abs_pos_emb=fm_args.use_abs_pos_emb, use_ch_emb=fm_args.use_ch_emb, use_mean_pooling=fm_args.use_mean_pooling, encoder_type=fm_args.enc_type, head_type=fm_args.head_type
    )
    
    vq_encoder = VectorQuantizeEncoder(
        seq_len=vq_args.seq_len, patch_size=vq_args.patch_size, in_channel=1, out_channel=vq_args.out_channel, num_class=0, num_group=vq_args.num_group,
        num_tokens=vq_args.num_tokens, d_codebook=vq_args.d_codebook, d_model=vq_args.d_model, num_head=vq_args.num_head, num_layer=vq_args.e_layers, qkv_bias=vq_args.qkv_bias, qk_scale=vq_args.qk_scale, qk_norm=vq_args.qk_norm,
        dropout=vq_args.dropout, attn_dropout=vq_args.attn_dropout, norm=vq_args.norm, activation=vq_args.activation, use_abs_pos_emb=vq_args.use_abs_pos_emb, use_ch_emb=vq_args.use_ch_emb, use_mean_pooling=vq_args.use_mean_pooling,
        decay=vq_args.ema_decay, eps=vq_args.eps, statistic_code_usage=vq_args.statistic_code_usage, kmeans_init=vq_args.kmeans_init, codebook_init_path=vq_args.codebook_init_path,
        encoder_type=vq_args.enc_type
    )

    device = f'cuda:{fm_args.rank}' if fm_args.use_ddp else fm_args.device
    eeg_encoder = eeg_encoder.to(device)
    vq_encoder = vq_encoder.to(device)
    vq_encoder = transfer_weights(fm_args.weights_path, vq_encoder, device, key='encoder_state_dict')
    
    model = EncoderSVREEGPretrain(fm_args, eeg_encoder, vq_encoder)
    model = model.to(device)

    n_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if fm_args.rank == 0:  # 0번 프로세스에서만 출력
        print(f'Number of learnable parameters: {n_learnable_params / 1e6} M')
        print(f'Number of fixed parameters: {n_fixed_params / 1e6} M')

    if fm_args.use_ddp:
        model = DDP(model, device_ids=[fm_args.rank], find_unused_parameters=False)



    return model

def get_dataset(args):
    base_dir = args.root_path
    
    """ 1. Load Dataset """
    # prepare pretrain_datasets, pretrain_datasets_time_window
    # time window for each sublist in dataset_train to ensure the total sequence length be around 256 for each dataset
    pretrain_datasets_dict = {
            'Mumtaz2016': 19,
            'SEED-IV': 62,
            'BCIC2020-3': 64,
            'DREAMER': 14,
            'SHU-MI': 32, 
            'Berlin_dsr': 26 , 
            'Berlin_nback': 26,
            'Berlin_wg': 26,   
            'Neonate': 19,
            'Fatigueset': 4,
            'Mental Arithmetic': 19,
            'Stew': 14,
            'TUSZ': 20,
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

    if args.rank == 0:
        print(pretrain_datasets_dict)
    pretrain_datasets = list(pretrain_datasets_dict.keys())
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

    if args.rank==0:  
        print(f"Total timestamps in pretraining datasets: {total_timestamps}")
        print(f'Total hours in pretraining datasets: {total_timestamps / (200 * 3600):.3f} hours')
        print(f"Data loading time: {(time.time() - data_load_start)//60} minutes")

    sampler_train_list = []
    for dataset in dataset_train_list:
        sampler_train = torch.utils.data.RandomSampler(dataset)
        sampler_train_list.append(sampler_train)

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            multiprocessing_context=mp.get_context('fork'), 
            # drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)

    return data_loader_train_list, train_ch_idx_list

def main(rank, world_size, vq_args, fm_args):
    if fm_args.use_ddp:
        setup_ddp(rank, world_size, fm_args.dist_url)
        fm_args.rank = rank
        fm_args.world_size = world_size
        fm_args.device = torch.device(f'cuda:{rank}')
        
    model = get_models(vq_args, fm_args)
    data_loader_train_list, train_ch_idx_list = get_dataset(fm_args)
    n_iters_per_epoch = sum([len(data_loader) for data_loader in data_loader_train_list])
    lr_schedule_values = cosine_scheduler(
                                        fm_args.lr, fm_args.min_lr, fm_args.train_epochs, n_iters_per_epoch,
                                        warmup_epochs=fm_args.warmup_epochs, warmup_lr_init=0
                                        )
    wd_schedule_values = cosine_scheduler(
                                        fm_args.weight_decay, fm_args.weight_decay, fm_args.train_epochs, n_iters_per_epoch,
                                        warmup_epochs=fm_args.warmup_epochs,
    )
    
    if rank == 0:
        print('Scheduler parameters:')
        print(f'  Initial learning rate: {fm_args.lr}')
        print(f'  Minimum learning rate: {fm_args.min_lr}')
        print(f'  Warmup epochs: {fm_args.warmup_epochs}')
        print(f'  Total epochs: {fm_args.train_epochs}')
        print(f'  Total iterations per epoch: {n_iters_per_epoch}')
        
    if rank == 0:
        print(fm_args)

    exp = Exp_SVREEG_Pretrain(fm_args, model, data_loader_train_list, train_ch_idx_list)
    exp.pretrain(lr_schedule_values, wd_schedule_values)

    if fm_args.use_ddp:
        cleanup_ddp()

if __name__ == '__main__':
    known, unknown = parser.parse_known_args()
    with open(known.vq_configs, 'r') as f:
        vq_args = yaml.safe_load(f)

    for key, value in vq_args.items():
        if isinstance(value, str):
            try: vq_args[key] = float(value)
            except ValueError: pass
    vq_args = argparse.Namespace(**vq_args)

    fm_cfgs = _load_yaml(known.fm_configs) 
    fm_parser = build_fm_pretrain_parser(defaults=fm_cfgs)
    fm_args = fm_parser.parse_args()

    set_seed(fm_args.seed)
    fm_args.num_patch = fm_args.seq_len // fm_args.patch_size

    fm_args.weights_path = os.path.join(fm_args.pretrain_path, f'quantizer', 'vq_ckpt_best.pth')
    fm_args.save_path = os.path.join(fm_args.pretrain_path, 'eegmodel', fm_args.model)
    if not os.path.exists(fm_args.save_path):
        os.makedirs(fm_args.save_path)

    if fm_args.use_ddp:
        fm_args.devices = fm_args.devices.replace(' ', '')
        device_ids = fm_args.devices.split(',')
        fm_args.world_size = len(device_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = fm_args.devices

        # If launched via torchrun / torch.distributed.launch, use env vars (recommended)
        local_rank_env = os.environ.get('LOCAL_RANK') or os.environ.get('SLURM_LOCALID') or os.environ.get('OMPI_COMM_WORLD_RANK')
        if local_rank_env is not None:
            # torchrun launched processes: call main directly per-process
            local_rank = int(local_rank_env)
            fm_args.rank = local_rank
            fm_args.world_size = int(os.environ.get('WORLD_SIZE', fm_args.world_size))
            # device assignment
            fm_args.device = torch.device(f'cuda:{fm_args.rank}')
            main(fm_args.rank, fm_args.world_size, vq_args, fm_args)
        else:
            # Fallback: spawn processes locally, force 'fork' start method to inherit open file handles
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass
            mp.spawn(main, args=(fm_args.world_size, vq_args, fm_args), nprocs=fm_args.world_size, join=True)

        # mp.spawn(main, args=(fm_args.world_size, vq_args, fm_args), nprocs=fm_args.world_size, join=True)

    else:
        if fm_args.use_gpu and fm_args.use_multi_gpu:
            fm_args.devices = fm_args.devices.replace(' ', '')
            device_ids = fm_args.devices.split(',')
            fm_args.device_ids = [int(id_) for id_ in device_ids]

        if not fm_args.use_gpu:
            fm_args.device = torch.device('cpu')
        else:
            if fm_args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = fm_args.devices
                print('Use Multi GPU: cuda:{}'.format(fm_args.devices))

            fm_args.device = torch.device('cuda:{}'.format(fm_args.gpu))
            if not fm_args.use_multi_gpu:
                print('Use GPU: cuda:{}'.format(fm_args.gpu))

        main(0, 1, vq_args, fm_args)
                
    