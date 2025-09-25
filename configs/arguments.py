import os
import yaml
import argparse

def _load_yaml(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def build_vq_pretrain_parser(defaults=None):
    """
    전체 인자 파서 생성. defaults(dict)를 parser.set_defaults로 적용하면
    --config로 불러온 yaml 값이 기본값이 되고 CLI가 우선합니다.
    """
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description='Noise-Robust Vector Quantizer Pre-training')
    
    parser.set_defaults(**defaults)
    # 최소한의 공통 인자들만 정의 (필요하면 추가)
    parser.add_argument('--vq_configs', type=str, default='configs/vector_quantizer/vq.yaml')
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 2025))
    # dataset / dataloader
    parser.add_argument('--root_path', type=str, default=defaults.get('root_path', '../dataset/processed/eeg'))
    parser.add_argument('--seq_len', type=int, default=defaults.get('seq_len', 25600))
    parser.add_argument('--patch_size', type=int, default=defaults.get('patch_size', 200))
    parser.add_argument('--num_workers', type=int, default=defaults.get('num_workers', 8))
    parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size', 256))
    parser.add_argument('--stride', type=int, default=defaults.get('stride', 200))

    # define vq encoder & decoder
    parser.add_argument('--out_channel', type=int, default=defaults.get('out_channel', 8))
    parser.add_argument('--e_layers', type=int, default=defaults.get('e_layers', 9))
    parser.add_argument('--d_layers', type=int, default=defaults.get('d_layers', 3))
    parser.add_argument('--d_model', type=int, default=defaults.get('d_model', 200))
    parser.add_argument('--num_head', type=int, default=defaults.get('num_head', 10))
    parser.add_argument('--num_group', type=int, default=defaults.get('num_group', 6))
    parser.add_argument('--qkv_bias', type=bool, default=defaults.get('qkv_bias', False))
    parser.add_argument('--qk_norm', type=str, default=defaults.get('qk_norm', None))
    parser.add_argument('--qk_scale', type=bool, default=defaults.get('qk_scale', None))
    parser.add_argument('--dropout', type=float, default=defaults.get('dropout', 0.1))
    parser.add_argument('--attn_dropout', type=float, default=defaults.get('attn_dropout', 0.1))
    parser.add_argument('--norm', type=str, default=defaults.get('norm', 'layernorm'))
    parser.add_argument('--activation', type=str, default=defaults.get('activation', 'gelu'))
    parser.add_argument('--use_abs_pos_emb', type=bool, default=defaults.get('use_abs_pos_emb', True))
    parser.add_argument('--use_ch_emb', type=bool, default=defaults.get('use_ch_emb', True))
    parser.add_argument('--use_mean_pooling', type=bool, default=defaults.get('use_mean_pooling', False))

    # define quantizer
    parser.add_argument('--num_tokens', type=int, default=defaults.get('num_tokens', 8192))
    parser.add_argument('--d_codebook', type=int, default=defaults.get('d_codebook', 64))
    parser.add_argument('--ema_decay', type=float, default=defaults.get('ema_decay', 0.99))
    parser.add_argument('--eps', type=float, default=defaults.get('eps', 1e-5))
    parser.add_argument('--statistic_code_usage', type=bool, default=defaults.get('statistic_code_usage', True))
    parser.add_argument('--kmeans_init', type=bool, default=defaults.get('kmeans_init', True))
    parser.add_argument('--codebook_init_path', type=str, default=defaults.get('codebook_init_path', ''))

    # define augmentation
    parser.add_argument('--noise_sample_ratio', type=float, default=defaults.get('noise_sample_ratio', 0.5))
    parser.add_argument('--noise_std', type=float, default=defaults.get('noise_std', 0.05))
    # training
    parser.add_argument('--train_epochs', type=int, default=defaults.get('train_epochs', 50))
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 5e-4))
    parser.add_argument('--min_lr', type=float, default=defaults.get('min_lr', 1e-5))
    parser.add_argument('--warmup_lr_init', type=float, default=defaults.get('warmup_lr_init', 1e-6))
    parser.add_argument('--warmup_epochs', type=int, default=defaults.get('warmup_epochs', 5))
    parser.add_argument('--loss', type=str, default=defaults.get('loss', 'l1_loss'))
    parser.add_argument('--weight_decay', type=float, default=defaults.get('weight_decay', 1e-4))
    parser.add_argument('--pretrain_path', type=str, default=defaults.get('pretrain_path', 'outputs/pretrain_checkpoints'))
    parser.add_argument('--max_grad_norm', type=float, default=defaults.get('max_grad_norm', 3.0))

    parser.add_argument('--start_epoch', type=int, default=defaults.get('start_epoch', 0))
    parser.add_argument('--save_freq', type=int, default=defaults.get('save_freq', 5))

    # gpu
    parser.add_argument('--use_gpu', type=bool, default=defaults.get('use_gpu', True))
    parser.add_argument('--gpu', type=int, default=defaults.get('gpu', 0))
    parser.add_argument('--use_multi_gpu', action='store_true', default=defaults.get('use_multi_gpu', False))
    parser.add_argument('--devices', type=str, default=defaults.get('devices', '0,1,2,3'))

    # DDP 관련 추가
    parser.add_argument('--use_ddp', action='store_true', default=defaults.get('use_ddp', False))
    parser.add_argument('--world_size', type=int, default=defaults.get('world_size', 1))
    parser.add_argument('--rank', type=int, default=defaults.get('rank', 0))
    parser.add_argument('--dist_url', type=str, default=defaults.get('dist_url', 'tcp://127.0.0.1:23456'))
    # parser.add_argument('--statistic_code_usage', type=bool, default=defaults.get('statistic_code_usage', False))  # True → False로 변경

    parser.add_argument('--vq_task', type=int, default=defaults.get('vq_task', 2))
    parser.add_argument('--enc_type', type=str, default=defaults.get('enc_type', 'epa2'))

    # wandb
    parser.add_argument('--use_wandb', action='store_true', default=defaults.get('use_wandb', False))
    parser.add_argument('--project', type=str, default=defaults.get('project', 'vq-pretraining'))
    parser.add_argument('--run_name', type=str, default=defaults.get('run_name', '0'))
    parser.add_argument('--wandb_dir', type=str, default=defaults.get('wandb_dir', './outputs/wandb'))
    # 필요한 추가 인자는 여기 더 추가 가능
    return parser

def build_fm_pretrain_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description='Subject-variability-robust EEG encoder Pre-training')
    parser.set_defaults(**defaults)

    parser.add_argument('--vq_configs', type=str, default=defaults.get('vq_configs', 'configs/vector_quantizer/vq.yaml'))
    parser.add_argument('--fm_configs', type=str, default=defaults.get('fm_configs', 'configs/eeg_fm/eegfm_base.yaml'))
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 2025))
    # dataset & dataloader
    parser.add_argument('--root_path', type=str, default=defaults.get('root_path', '../dataset/processed/eeg'))
    parser.add_argument('--seq_len', type=int, default=defaults.get('seq_len', 25600))
    parser.add_argument('--patch_size', type=int, default=defaults.get('patch_size', 200))
    parser.add_argument('--num_workers', type=int, default=defaults.get('num_workers', 8))
    parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size', 256))
    parser.add_argument('--stride', type=int, default=defaults.get('stride', 800))

    # define vq encoder & decoder
    parser.add_argument('--out_channel', type=int, default=defaults.get('out_channel', 8))
    parser.add_argument('--e_layers', type=int, default=defaults.get('e_layers', 4))
    parser.add_argument('--d_layers', type=int, default=defaults.get('d_layers', 2))
    parser.add_argument('--d_model', type=int, default=defaults.get('d_model', 200))
    parser.add_argument('--num_head', type=int, default=defaults.get('num_head', 10))
    parser.add_argument('--num_group', type=int, default=defaults.get('num_group', 6))
    parser.add_argument('--qkv_bias', type=bool, default=defaults.get('qkv_bias', False))
    parser.add_argument('--qk_norm', type=str, default=defaults.get('qk_norm', None))
    parser.add_argument('--qk_scale', type=bool, default=defaults.get('qk_scale', None))
    parser.add_argument('--dropout', type=float, default=defaults.get('dropout', 0.1))
    parser.add_argument('--attn_dropout', type=float, default=defaults.get('attn_dropout', 0.1))
    parser.add_argument('--norm', type=str, default=defaults.get('norm', 'layernorm'))
    parser.add_argument('--activation', type=str, default=defaults.get('activation', 'gelu'))
    parser.add_argument('--use_abs_pos_emb', type=bool, default=defaults.get('use_abs_pos_emb', True))
    parser.add_argument('--use_ch_emb', type=bool, default=defaults.get('use_ch_emb', True))
    parser.add_argument('--use_mean_pooling', type=bool, default=defaults.get('use_mean_pooling', False))
    parser.add_argument('--head_type', type=str, default=defaults.get('head_type', 'pretrain'))
    parser.add_argument('--mask_prob', type=float, default=defaults.get('mask_prob', 0.5))

    # define quantizer
    parser.add_argument('--num_tokens', type=int, default=defaults.get('num_tokens', 8192))
    parser.add_argument('--d_codebook', type=int, default=defaults.get('d_codebook', 64))
    parser.add_argument('--decay', type=float, default=defaults.get('decay', 0.99))
    parser.add_argument('--eps', type=float, default=defaults.get('eps', 1e-5))
    parser.add_argument('--statistic_code_usage', type=bool, default=defaults.get('statistic_code_usage', True))
    parser.add_argument('--kmeans_init', type=bool, default=defaults.get('kmeans_init', True))
    parser.add_argument('--codebook_init_path', type=str, default=defaults.get('codebook_init_path', ''))

    # training
    parser.add_argument('--train_epochs', type=int, default=defaults.get('train_epochs', 50))
    parser.add_argument('--lr', type=float, default=defaults.get('lr', 1e-4))
    parser.add_argument('--min_lr', type=float, default=defaults.get('min_lr', 1e-5))
    parser.add_argument('--warmup_lr_init', type=float, default=defaults.get('warmup_lr_init', 1e-6))
    parser.add_argument('--warmup_epochs', type=int, default=defaults.get('warmup_epochs', 5))
    parser.add_argument('--loss', type=str, default=defaults.get('loss', 'l1_loss'))
    parser.add_argument('--pretrain_path', type=str, default=defaults.get('pretrain_path', 'outputs/pretrain_checkpoints'))
    parser.add_argument('--max_grad_norm', type=float, default=defaults.get('max_grad_norm', 3.0))
    parser.add_argument('--weight_decay', type=float, default=defaults.get('weight_decay', 5e-2))

    parser.add_argument('--start_epoch', type=int, default=defaults.get('start_epoch', 0))
    parser.add_argument('--save_freq', type=int, default=defaults.get('save_freq', 10))

    # gpu
    parser.add_argument('--use_gpu', type=bool, default=defaults.get('use_gpu', True))
    parser.add_argument('--gpu', type=int, default=defaults.get('gpu', 0))
    parser.add_argument('--use_multi_gpu', action='store_true', default=defaults.get('use_multi_gpu', False))
    parser.add_argument('--devices', type=str, default=defaults.get('devices', '0,1,2,3'))

    # DDP 관련 추가
    parser.add_argument('--use_ddp', action='store_true', default=defaults.get('use_ddp', False))
    parser.add_argument('--world_size', type=int, default=defaults.get('world_size', 1))
    parser.add_argument('--rank', type=int, default=defaults.get('rank', 0))
    parser.add_argument('--dist_url', type=str, default=defaults.get('dist_url', 'tcp://127.0.0.1:23456'))

    # vq task version
    parser.add_argument('--vq_task', type=int, default=defaults.get('vq_task', 2))
    parser.add_argument('--enc_type', type=str, default=defaults.get('enc_type', 'epa2'))

    # wandb
    parser.add_argument('--use_wandb', action='store_true', default=defaults.get('use_wandb', False))
    parser.add_argument('--project', type=str, default=defaults.get('project', 'fm-pretraining'))
    parser.add_argument('--run_name', type=str, default=defaults.get('run_name', '0'))
    parser.add_argument('--wandb_dir', type=str, default=defaults.get('wandb_dir', './outputs/wandb'))

    return parser

def build_fm_finetune_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description='Subject-variability-robust EEG encoder Pre-training')
    parser.set_defaults(**defaults)

    parser.add_argument('--vq_configs', type=str, default='configs/vector_quantizer/vq.yaml')
    parser.add_argument('--fm_configs', type=str, default='configs/eeg_fm/eegfm_base.yaml')
    parser.add_argument('--seed', type=int, default=2025)
    # dataset & dataloader
    parser.add_argument('--root_path', type=str, default='../dataset/processed/eeg')
    parser.add_argument('--seq_len', type=int, default=25600)
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=800)

    # define vq encoder & decoder
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--num_head', type=int, default=10)
    parser.add_argument('--num_group', type=int, default=5)
    parser.add_argument('--qkv_bias', type=bool, default=False)
    parser.add_argument('--qk_norm', type=str, default=None)
    parser.add_argument('--qk_scale', type=bool, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='layernorm')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--use_abs_pos_emb', type=bool, default=True)
    parser.add_argument('--use_ch_emb', type=bool, default=True)
    parser.add_argument('--use_mean_pooling', type=bool, default=False)
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)

    # define quantizer
    parser.add_argument('--num_tokens', type=int, default=8192)
    parser.add_argument('--d_codebook', type=int, default=64)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--statistic_code_usage', type=bool, default=True)
    parser.add_argument('--kmeans_init', type=bool, default=True)
    parser.add_argument('--codebook_init_path', type=str, default='')

    # training
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_lr_init', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--loss', type=str, default='l1_loss')
    parser.add_argument('--pretrain_path', type=str, default='outputs/pretrain_checkpoints')
    parser.add_argument('--max_grad_norm', type=float, default=3.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=10)

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
    parser.add_argument('--project', type=str, default='fm-pretraining')
    parser.add_argument('--run_name', type=str, default='0')
    parser.add_argument('--wandb_dir', type=str, default='./outputs/wandb')

    return parser