import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import GradScaler

import time
import os

from einops import rearrange, repeat
from collections import OrderedDict

import wandb

class Exp_SVREEG_Pretrain:
    def __init__(self, args, eeg_model, data_loader_list, ch_idx_list):
        self.args = args
        self.eeg_model = eeg_model
        self.data_loader_list = data_loader_list
        self.ch_idx_list = ch_idx_list
        self.n_iters_per_epoch = sum([len(data_loader) for data_loader in self.data_loader_list])
        
        # DDP 환경 체크
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.device = args.device
        
        self.ckpt_dir = args.save_path
        # Initialize wandb only on rank 0 to avoid duplicate runs in DDP
        if (self.rank == 0) and args.use_wandb:
            wandb.init(
                project=args.project,
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )
            
    def _select_optimizer(self):
        optimizer = torch.optim.AdamW(
                                    self.eeg_model.parameters(), lr=self.args.lr, 
                                    weight_decay=self.args.weight_decay
                                    )
        return optimizer

    def save_checkpoint(self, epoch, loss_dict, ckpt_path, save_best_this_epoch=False):
        # rank 0에서만 저장
        if self.rank != 0:
            return

        if loss_dict is not None:
            # DDP와 DataParallel 호환
            if hasattr(self.eeg_model, 'module'):
                model = self.eeg_model.module
            else:
                model = self.eeg_mode
                
            # Extract encoder and decoder state dicts
            encoder_state_dict = OrderedDict()
            for k, v in model.state_dict().items():
                if 'vq_encoder' not in k:
                    k = k.replace('encoder.', '')
                    encoder_state_dict[k] = v

            # Save checkpoint
            ckpt = {
                'epoch': epoch,
                'loss': loss_dict,
                'best_loss': self.min_loss,
                'encoder_state_dict': encoder_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict()
            }
            torch.save(ckpt, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')
            
            # true라면 현재 저장한 모델은 best model
            if save_best_this_epoch:
                self.best_ckpt = ckpt
        
        else:
            # 현재 epoch까지의 best model 저장
            torch.save(self.best_ckpt, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')

    def load_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.ckpt_dir, f'encoder_ckpt_ep{epoch-1}.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
            
        ckpt = torch.load(ckpt_path, map_location=self.args.device)
        if hasattr(self.eeg_model, 'module'):
            model = self.eeg_model.module
        else:
            model = self.eeg_model
        model.load_state_dict(ckpt['encoder_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.min_loss = ckpt['best_loss']
        if self.rank==0:
            print(f'Loaded checkpoint from {ckpt_path}')

    def pretrain(self, lr_schedule_values, wd_schedule_values):
        self.optimizer = self._select_optimizer()
        self.scaler = GradScaler()
        self.lr_schedule_values = lr_schedule_values
        self.wd_schedule_values = wd_schedule_values
        self.min_loss = 1e9
        
        if self.args.start_epoch != 0:
            if self.rank == 0:
                print(f'Resuming training from epoch {self.args.start_epoch}')
            self.load_checkpoint(self.args.start_epoch) 

        for epoch in range(self.args.start_epoch, self.args.train_epochs):
            # DDP에서 각 epoch마다 sampler 설정
            if hasattr(self.data_loader_list[0], 'sampler') and hasattr(self.data_loader_list[0].sampler, 'set_epoch'):
                for data_loader in self.data_loader_list:
                    data_loader.sampler.set_epoch(epoch)
            
            start_time = time.time()
            loss, koleo_loss, codebook_loss = self.pretrain_one_epoch(epoch)
            end_time = time.time()
            
            if self.is_ddp:
                metrics = torch.tensor([loss, koleo_loss, codebook_loss], device=self.device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics = metrics / self.world_size
                loss, koleo_loss, codebook_loss = metrics.cpu().numpy()
                
            if self.rank==0:
                print(f'Epoch: {epoch+1}/{self.args.train_epochs}, '
                    f'Loss: {loss:.4f}, Koleo Loss: {koleo_loss:.4f}, '
                    f'Codebook Loss: {codebook_loss:.4f}, '
                    f'Time: {end_time-start_time:.2f}s')

                loss_dict = {
                    'loss': loss,
                    'koleo_loss': koleo_loss,
                    'codebook_loss': codebook_loss
                }
                if wandb.run is not None:
                    wandb.log(
                        loss_dict, 
                        # step=epoch
                        )
                
                # Save best model
                if loss < self.min_loss:
                    print(f'Pre-training loss decreased ({self.min_loss:.4f} --> {loss:.4f}). '
                        f'Saving model epoch {epoch}...')
                    self.min_loss = loss
                    best_ckpt_path = os.path.join(self.ckpt_dir, 'encoder_ckpt_best.pth')
                    self.save_checkpoint(epoch, loss_dict, best_ckpt_path, save_best_this_epoch=True)
                    
                # Save periodic checkpoint
                if self.args.save_freq > 0 and (epoch+1) % self.args.save_freq == 0:
                    periodic_ckpt_path = os.path.join(self.ckpt_dir, f'encoder_ckpt_ep{epoch}.pth')
                    self.save_checkpoint(epoch, loss_dict, periodic_ckpt_path, save_best_this_epoch=False)
                    best_ckpt_path = os.path.join(self.ckpt_dir, f'encoder_ckpt_best_ep{epoch}.pth')
                    self.save_checkpoint(epoch, None, best_ckpt_path, save_best_this_epoch=True)


            # DDP 동기화 (모든 프로세스가 epoch 완료를 기다림)
            if self.is_ddp:
                dist.barrier()
                
    def pretrain_one_epoch(self, epoch):
        
        total_loss = total_koleo_loss = total_codebook_loss = 0

        total_steps = 0
        self.eeg_model.train()
        dset_count = 0
        for data_loader, input_chans_idx in zip(self.data_loader_list, self.ch_idx_list):
            dset_count +=1 
            load_start = time.time()
            for step, x in enumerate(data_loader):
                if (self.rank == 0) & (step == 0):
                    print(f'dset {dset_count}/{len(self.data_loader_list)} n_iters: {len(data_loader)} [T: {time.time()-load_start:.1f}]: (ch: {len(input_chans_idx)}) total steps: {total_steps}')

                it = self.n_iters_per_epoch * epoch + step + total_steps
                if it < len(self.lr_schedule_values):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_schedule_values[it] * param_group.get('lr_scale', 1.0)
                        if self.wd_schedule_values is not None and param_group['weight_decay'] > 0:
                            param_group['weight_decay'] = self.wd_schedule_values[it]
                else:
                    print(f'Warning: it ({it}) exceeds lr_schedule_values length ({len(self.lr_schedule_values)}). Using last value.')
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_schedule_values[-1] * param_group.get('lr_scale', 1.0)
                        if self.wd_schedule_values is not None and param_group['weight_decay'] > 0:
                            param_group['weight_decay'] = self.wd_schedule_values[-1]
                            
                if it == len(self.lr_schedule_values)-1:
                    print('마지막 학습률까지 돌았음')

                batch_size, c, seq_len = x.shape
                x = x.float().to(self.args.device)

                # Model forward pass
                with torch.amp.autocast('cuda'):
                    loss, koleo_loss, codebook_loss = self.eeg_model(x, input_chans_idx)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.eeg_model.parameters(), max_norm=self.args.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Accumulate losses
                total_loss += loss.item()
                total_koleo_loss += koleo_loss.item()
                total_codebook_loss += codebook_loss.item()

                # Log step metrics to wandb (only on rank 0)
                if self.rank == 0 and wandb.run is not None:
                    wandb.log({
                        'step/loss': loss.item(),
                        'step/koleo_loss': koleo_loss.item(),
                        'step/codebook_loss': codebook_loss.item(),
                        'step': it
                    }, step=it)

            total_steps += len(data_loader)
        
        # Calculate average losses
        total_loss /= self.n_iters_per_epoch
        total_koleo_loss /= self.n_iters_per_epoch
        total_codebook_loss /= self.n_iters_per_epoch

        return total_loss, total_koleo_loss, total_codebook_loss

