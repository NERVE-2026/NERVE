#%%


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time
import os
import random

from einops import rearrange, repeat
from collections import OrderedDict
import wandb
from torch.cuda.amp import GradScaler
from utils import transfer_weights

#%%

#%%
class Exp_NRVQ:
    def __init__(self, args, vq_backbone, data_loader_list, ch_idx_list):
        self.args = args
        self.vq_backbone = vq_backbone
        self.data_loader_list = data_loader_list
        self.ch_idx_list = ch_idx_list
        self.n_iters_per_epoch = sum(len(data_loader) for data_loader in self.data_loader_list)
        
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
        optimizer = torch.optim.AdamW(self.vq_backbone.parameters(), lr=self.args.lr, betas=(0.9, 0.999),
                                      weight_decay=self.args.weight_decay)
        return optimizer

    def save_checkpoint(self, epoch, loss_dict, ckpt_path, save_best_this_epoch=False):
        if self.rank != 0:
            return

        if loss_dict is not None:
            if hasattr(self.vq_backbone, 'module'):
                model = self.vq_backbone.module
            else:
                model = self.vq_backbone
                
            encoder_state_dict = OrderedDict()
            decoder_state_dict = OrderedDict()
            for k, v in model.encoder.state_dict().items():
                encoder_state_dict[k] = v
            for k, v in model.decoder.state_dict().items():
                decoder_state_dict[k] = v

            # Save checkpoint
            ckpt = {
                'epoch': epoch,
                'loss': loss_dict,
                'best_loss': self.min_loss,
                'encoder_state_dict': encoder_state_dict,
                'decoder_state_dict': decoder_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict()
            }
            torch.save(ckpt, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')
            
            if save_best_this_epoch:
                self.best_ckpt = ckpt
        
        else:
            torch.save(self.best_ckpt, ckpt_path)
            print(f'Checkpoint saved to {ckpt_path}')

            

    def load_checkpoint(self, epoch):
        # ckpt_path = os.path.join(self.ckpt_dir, f'vq_ckpt_ep{epoch-1}.pth')
        ckpt_path = os.path.join(self.ckpt_dir, 'vq_ckpt_best.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found at {ckpt_path}')
            
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if hasattr(self.vq_backbone, 'module'):
            model = self.vq_backbone.module
        else:
            model = self.vq_backbone
        model.encoder.load_state_dict(ckpt['encoder_state_dict'])
        model.decoder.load_state_dict(ckpt['decoder_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.min_loss = ckpt['best_loss']
        
        if self.rank == 0:
            print(f'Loaded checkpoint from {ckpt_path}')

    def pretrain(self, lr_schedule_values):
        self.optimizer = self._select_optimizer()
        self.scaler = GradScaler()
        self.lr_schedule_values = lr_schedule_values
        self.min_loss = 1e9

        if self.args.start_epoch != 0:
            if self.rank == 0:
                print(f'Resuming training from epoch {self.args.start_epoch}')
            self.load_checkpoint(self.args.start_epoch)

        for epoch in range(self.args.start_epoch, self.args.train_epochs):
            
            if hasattr(self.data_loader_list[0], 'sampler') and hasattr(self.data_loader_list[0].sampler, 'set_epoch'):
                for data_loader in self.data_loader_list:
                    data_loader.sampler.set_epoch(epoch)

            model = self.vq_backbone.module if hasattr(self.vq_backbone, 'module') else self.vq_backbone
            model.encoder.quantizer.reset_epoch_codebook_usage()
            start_time = time.time()
            loss, emb_loss, temp_loss, freq_loss = self.pretrain_one_epoch(epoch)
            train_time = time.time() - start_time
            unused_codebook_count = model.encoder.quantizer.get_unused_codebook_count_epoch()
            codebook_usage_entropy = model.encoder.quantizer.get_codebook_usage_entropy_epoch()
            
            if self.is_ddp:
                metrics = torch.tensor([loss, emb_loss, temp_loss, freq_loss, unused_codebook_count, codebook_usage_entropy],
                                     device=self.device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics = metrics / self.world_size
                loss, emb_loss, temp_loss, freq_loss, unused_codebook_count, codebook_usage_entropy = metrics.cpu().numpy()
                unused_codebook_count = int(unused_codebook_count)
            
            if self.rank == 0:
                print(f'Epoch: {epoch+1}/{self.args.train_epochs}, '
                      f'Loss: {loss:.4f}, Emb Loss: {emb_loss:.4f}, '
                      f'Temp Loss: {temp_loss:.4f}, Freq Loss: {freq_loss:.5f}, '
                      f'Time: {train_time:.2f}s, '
                      f'Unused Codebook Count: {unused_codebook_count}/{self.args.num_tokens}, '
                      f'Codebook Usage Entropy: {codebook_usage_entropy:.4f}')


                loss_dict = {
                    'loss': loss,
                    'emb_loss': emb_loss,
                    'temp_loss': temp_loss,
                    'freq_loss': freq_loss,
                    'unused_codebook_count': unused_codebook_count,
                    'codebook_usage_entropy': codebook_usage_entropy,
                }
                        
                if wandb.run is not None:
                    wandb.log(
                            loss_dict, 
                              )
                
                if loss < self.min_loss:
                    print(f'Loss decreased ({self.min_loss:.4f} -> {loss:.4f}). Saving best model...')
                    self.min_loss = loss
                    best_ckpt_path = os.path.join(self.ckpt_dir, 'vq_ckpt_best.pth')
                    self.save_checkpoint(epoch, loss_dict, best_ckpt_path, save_best_this_epoch=True)

                # Save periodic checkpoint
                if self.args.save_freq > 0 and (epoch+1) % self.args.save_freq == 0:
                    periodic_ckpt_path = os.path.join(self.ckpt_dir, f'vq_ckpt_ep{epoch}.pth')
                    self.save_checkpoint(epoch, loss_dict, periodic_ckpt_path, save_best_this_epoch=False)
                    best_ckpt_path = os.path.join(self.ckpt_dir, f'vq_ckpt_best_ep{epoch}.pth')
                    self.save_checkpoint(epoch, None, best_ckpt_path, save_best_this_epoch=True)

            if self.is_ddp:
                dist.barrier()

            torch.cuda.empty_cache()

    def pretrain_one_epoch(self, epoch):
        total_loss = total_emb_loss = total_rec_loss_temp = total_rec_loss_freq = 0
        
        total_steps = 0
        self.vq_backbone.train()
        dset_count = 0
        for data_loader, input_ch_idx in zip(self.data_loader_list, self.ch_idx_list):
            dset_count += 1           
            load_start = time.time()   
            for step, x in enumerate(data_loader):
                if (self.rank == 0) & (step == 0):
                    print(f'Processing dataset {dset_count}/{len(self.data_loader_list)} [time: {time.time()-load_start:.1f}]: (input_ch_idx: {len(input_ch_idx)})')

                it = self.n_iters_per_epoch * epoch + step + total_steps
                if it<len(self.lr_schedule_values):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_schedule_values[it] * param_group.get('lr_scale', 1.0)
                else:
                    print(f'Warning: it ({it}) exceeds lr_schedule_values length ({len(self.lr_schedule_values)}). Using last value.')
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_schedule_values[-1] * param_group.get('lr_scale', 1.0)


                batch_size, c, seq_len = x.shape
                x = x.float().to(self.device)
                with torch.amp.autocast('cuda'):
                    emb_loss, rec_loss_temp, rec_loss_freq = self.vq_backbone(x, input_ch_idx)

                loss = emb_loss + rec_loss_temp + rec_loss_freq
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vq_backbone.parameters(), max_norm=self.args.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update metrics
                total_loss += loss.item()
                total_emb_loss += emb_loss.item()
                total_rec_loss_temp += rec_loss_temp.item()
                total_rec_loss_freq += rec_loss_freq.item()
                
                if self.rank == 0 and wandb.run is not None:
                    wandb.log({
                        'step/loss': loss.item(),
                        'step/emb_loss': emb_loss.item(),
                        'step/temp_loss': rec_loss_temp.item(),
                        'step/freq_loss': rec_loss_freq.item(),
                        'step/epoch': epoch,
                        'step/step': it,
                        'step/lr': self.optimizer.param_groups[0]['lr'],
                    },)
                # ---------------------------------
            total_steps += len(data_loader)

        # Calculate average losses
        total_loss /= self.n_iters_per_epoch
        total_emb_loss /= self.n_iters_per_epoch
        total_rec_loss_temp /= self.n_iters_per_epoch
        total_rec_loss_freq /= self.n_iters_per_epoch

        return total_loss, total_emb_loss, total_rec_loss_temp, total_rec_loss_freq
