#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

import time
import os
import numpy as np

import wandb
from collections import OrderedDict
from utils import Evaluator, ddp_reduce


#%%
class Exp_SVREEG_Finetune:
    def __init__(self, args, eeg_encoder, data_loader, ch_idx):
        self.args = args
        self.eeg_encoder = eeg_encoder
        self.evaluator = Evaluator()
        self.train_loader = data_loader['train']
        self.val_loader = data_loader['val']
        self.test_loader = data_loader['test']
        self.ch_idx = ch_idx
        self.n_iters_per_epoch = len(self.train_loader)
        self.criterion = None

        # DDP 환경 체크
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self.device = args.device
        
        # Initialize paths 
        self.best_model_path = os.path.join(args.task_path, 'best_model.pth')

        # Initialize wandb only on rank 0 to avoid duplicate runs in DDP
        if (self.rank == 0) and args.use_wandb:
            wandb.init(
                project=args.project,
                name=args.run_name,
                dir=args.wandb_dir,
                config=vars(args)
            )

    def _get_criterion(self):
        if self.args.num_class == 0:
            self.criterion = nn.MSELoss().cuda()
        elif self.args.num_class == 2: 
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).cuda()

    def _get_optimizer(self):
        # 학습이 잘 안되면, backbone과 head에 다른 learning rate 적용해보기 (CBRAMOD 창조)
        optimizer = torch.optim.AdamW(self.eeg_encoder.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def train(self, lr_schedule_values):
        self.optimizer = self._get_optimizer()
        self.lr_schedule_values = lr_schedule_values
        self._get_criterion()

        if self.args.num_class > 2:
            self.train_for_multiclass()
        elif self.args.num_class == 0:
            self.train_for_regression()
        else:
            self.train_for_binaryclass()
    
    def train_for_regression(self):
        best_metrics = {'corr': 0, 'r2': 0, 'rmse': 0}
        
        for epoch in range(self.args.train_epochs):
            # DDP에서 각 epoch마다 sampler 설정
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            train_loss = self.train_one_epoch_regression(self.train_loader, epoch, train=True)
            val_loss, val_corr, val_r2, val_rmse = self.train_one_epoch_regression(self.val_loader, train=False)
            test_loss, test_corr, test_r2, test_rmse = self.train_one_epoch_regression(self.test_loader, train=False)

            if self.is_ddp:
                metrics = [train_loss, val_loss, val_corr, val_r2, val_rmse]
                train_loss, val_loss, val_corr, val_r2, val_rmse = ddp_reduce(metrics, self.world_size, self.device)
                metrics = [test_loss, test_corr, test_r2, test_rmse]
                test_loss, test_corr, test_r2, test_rmse = ddp_reduce(metrics, self.world_size, self.device)

            if self.rank == 0:
                # Print epoch results
                print(f'Epoch: {epoch+1}/{self.args.train_epochs} ({time.time()-start_time:.2f}s)')
                print(f'[Train] Loss: {train_loss:.4f}')
                print(f'[Val]   Loss: {val_loss:.4f}, Corr: {val_corr:.4f}, R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}')
                print(f'[Test]  Loss: {test_loss:.4f}, Corr: {test_corr:.4f}, R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}')
                
                loss_dict = {
                    'train/loss': train_loss,
                    'val/loss': val_loss, 'val/corr': val_corr, 'val/r2': val_r2, 'val/rmse': val_rmse,
                    'test/loss': test_loss, 'test/corr': test_corr, 'test/r2': test_r2, 'test/rmse': test_rmse
                }
                
                if wandb.run is not None:
                    wandb.log(loss_dict)

                # Save best model
                if val_corr > best_metrics['corr']:
                    best_metrics.update({
                        'corr': val_corr,
                        'r2': val_r2,
                        'rmse': val_rmse
                    })

                    print(f'Saving best model... [val] Corr: {val_corr:.4f}, R2: {val_r2:.4f}, RMSE: {val_rmse:.4f}, [test] Corr: {test_corr:.4f}, R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}')

                    torch.save({
                        'epoch': epoch,
                        'val_res': {'corr': val_corr, 'r2': val_r2, 'rmse': val_rmse,},
                        'test_res': {'corr': test_corr, 'r2': test_r2, 'rmse': test_rmse,},
                        'encoder_state_dict': self.eeg_encoder.state_dict()
                    }, self.best_model_path)
                    
        if self.is_ddp:
            dist.barrier()
    
    def train_one_epoch_regression(self, data_loader, epoch=None, train=True):
        self.eeg_encoder.train() if train else self.eeg_encoder.eval()

        loss_log = []
        preds, targets = [], []
        
        for step, (x, y) in enumerate(data_loader):
            # Update learning rate
            if (epoch is not None) and train:
                it = self.n_iters_per_epoch * epoch + step
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule_values[it]

            # Forward pass
            x = x.float().to(self.args.device)
            y = y.to(self.args.device)
            pred = self.eeg_encoder(x, self.ch_idx)
            loss = self.criterion(pred.squeeze(), y.float())

            # Training step
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation step
            else:
                preds += pred.detach().cpu().squeeze().numpy().tolist()
                targets += y.cpu().squeeze().numpy().tolist()
            
            loss_log.append(loss.data.cpu().numpy())
        
        if train:
            return np.mean(loss_log)
        
        # Calculate metrics
        targets = np.array(targets)
        preds = np.array(preds)
        corr, r2, rmse = self.evaluator.get_regression_metrics(targets, preds)
        
        return np.mean(loss_log), corr, r2, rmse
    
    def train_for_multiclass(self):
        best_metrics = {'f1': 0, 'kappa': 0, 'acc': 0, 'cm': None}
        
        for epoch in range(self.args.train_epochs):
            # DDP에서 각 epoch마다 sampler 설정
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            # Training phase
            train_loss = self.train_one_epoch_multiclass(self.train_loader, epoch, train=True)
            val_loss, val_acc, val_f1, val_kappa, val_cm = self.train_one_epoch_multiclass(self.val_loader, train=False)
            test_loss, test_acc, test_f1, test_kappa, test_cm = self.train_one_epoch_multiclass(self.test_loader, train=False)

            # print(val_cm)
            # print(test_cm)
            
            if self.is_ddp:
                # train, validation
                metrics = [train_loss, val_loss, val_acc, val_f1, val_kappa]
                train_loss, val_loss, val_acc, val_f1, val_kappa = ddp_reduce(metrics, self.world_size, self.device)
                val_cm = ddp_reduce(val_cm, 1, self.device)

                # test
                metrics = [test_loss, test_acc, test_f1, test_kappa]
                test_loss, test_acc, test_f1, test_kappa = ddp_reduce(metrics, self.world_size, self.device)
                test_cm = ddp_reduce(test_cm, 1, self.device)

            if self.rank==0:
                # Print epoch results
                print(f'Epoch: {epoch+1}/{self.args.train_epochs} ({time.time()-start_time:.2f}s)')
                print(f'[Train] Loss: {train_loss:.4f}')
                print(f'[Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}')
                print(f'[Test]  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Kappa: {test_kappa:.4f}')

                loss_dict = {
                    'train/loss': train_loss,
                    'val/loss': val_loss, 'val/acc': val_acc, 'val/f1': val_f1, 'val/kappa': val_kappa,
                    'test/loss': test_loss, 'test/acc': test_acc, 'test/f1': test_f1, 'test/kappa': test_kappa
                }

                if wandb.run is not None:
                    wandb.log(loss_dict)
                    
                # Save best model
                if val_kappa > best_metrics['kappa']:
                    best_metrics.update({
                        'kappa': val_kappa,
                        'acc': val_acc,
                        'f1': val_f1,
                        'cm': val_cm
                    })

                    print(f'Saving best model... [Val] Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}, F1: {val_f1:.4f}, [test] Acc: {test_acc:.4f}, Kappa: {test_kappa:.4f}, F1: {test_f1:.4f}')

                    torch.save({
                        'epoch': epoch,
                        'val_res': {'acc': val_acc, 'kappa': val_kappa, 'f1': val_f1, 'cm': val_cm},
                        'test_res': {'acc': test_acc, 'kappa': test_kappa, 'f1': test_f1, 'cm': test_cm},
                        'encoder_state_dict': self.eeg_encoder.state_dict()
                    }, self.best_model_path)
                    
        if self.is_ddp:
            dist.barrier()

    def train_one_epoch_multiclass(self, data_loader, epoch=None, train=True):
        self.eeg_encoder.train() if train else self.eeg_encoder.eval()

        loss_log = []
        preds, targets = [], []
        
        for step, (x, y) in enumerate(data_loader):
            # Update learning rate
            if (epoch is not None) and train:
                it = self.n_iters_per_epoch * epoch + step
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule_values[it]

            # Forward pass
            x = x.float().to(self.args.device)
            y = y.to(self.args.device)
            pred = self.eeg_encoder(x, self.ch_idx)
            loss = self.criterion(pred, y.long())

            # Training step
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # Validation step
            else:
                pred = pred.detach()
                pred = torch.max(pred, dim=-1)[1]
                preds += pred.cpu().squeeze().numpy().tolist()
                targets += y.cpu().squeeze().numpy().tolist()
            
            loss_log.append(loss.data.cpu().numpy())
        
        if not train:
            from collections import Counter
            print(Counter(targets))

        if train:
            return np.mean(loss_log)
        
        # Calculate metrics
        targets = np.array(targets)
        preds = np.array(preds)
        acc, f1, kappa, cm = self.evaluator.get_multiclass_metrics(targets, preds)
        metrics = {
            'loss': np.mean(loss_log),
            'acc': acc,
            'f1': f1,
            'kappa': kappa,
            'cm': cm
        }
        return np.mean(loss_log), acc, f1, kappa, cm
        
    def train_for_binaryclass(self):
        best_metrics = {'acc': 0, 'auprc':0, 'auc': 0, 'prec': 0, 'recall': 0, 'cm': None}

        for epoch in range(self.args.train_epochs):
            # DDP에서 각 epoch마다 sampler 설정
            if self.is_ddp:
                self.train_loader.sampler.set_epoch(epoch)

            start_time = time.time()
            train_loss = self.train_one_epoch_binaryclass(self.train_loader, epoch, train=True)
            val_loss, val_acc, val_auprc, val_auc, val_prec, val_recall, val_cm = self.train_one_epoch_binaryclass(self.val_loader, train=False)
            test_loss, test_acc, test_auprc, test_auc, test_prec, test_recall, test_cm = self.train_one_epoch_binaryclass(self.test_loader, train=False)
            
            if self.is_ddp:
                # train, validation 
                metrics = [train_loss, val_loss, val_acc, val_auprc, val_auc, val_prec, val_recall]
                train_loss, val_loss, val_acc, val_auprc, val_auc, val_prec, val_recall = ddp_reduce(metrics, self.world_size, self.device)
                val_cm = ddp_reduce(val_cm, 1, self.device)

                # test
                metrics = [test_loss, test_acc, test_auprc, test_auc, test_prec, test_recall]
                test_loss, test_acc, test_auprc, test_auc, test_prec, test_recall = ddp_reduce(metrics, self.world_size, self.device)
                test_cm = ddp_reduce(test_cm, 1, self.device)
            
            if self.rank==0:
                # Print epoch results
                print(f'Epoch: {epoch+1}/{self.args.train_epochs} ({time.time()-start_time:.2f}s)')
                print(f'[Train] Loss: {train_loss:.4f}')
                print(f'[Val]   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUPRC: {val_auprc:.4f} AUC: {val_auc:.4f}, Prec: {val_prec:.4f}, Recall: {val_recall:.4f}')
                print(f'[Test]  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUPRC: {test_auprc:.4f} AUC: {test_auc:.4f}, Prec: {test_prec:.4f}, Recall: {test_recall:.4f}')

                loss_dict = {
                    'train/loss': train_loss,
                    'val/loss': val_loss, 'val/acc': val_acc, 'val/auprc': val_auprc, 'val/auc': val_auc, 'val/prec': val_prec, 'val/recall': val_recall,
                    'test/loss': test_loss, 'test/acc': test_acc, 'test/auprc': test_auprc, 'test/auc': test_auc, 'test/prec': test_prec, 'test/recall': test_recall
                }

                if wandb.run is not None:
                    wandb.log(loss_dict)

                # Save best model
                if val_auc > best_metrics['auc']:
                    best_metrics.update({
                        'auc': val_auc,
                        'auprc': val_auprc,
                        'acc': val_acc,
                        'prec': val_prec,
                        'recall': val_recall,
                        'cm': val_cm
                    })

                    print(f'Saving best model... [val] Acc: {val_acc:.4f}, AUPRC: {val_auprc:.4f} AUC: {val_auc:.4f}, Prec: {val_prec:.4f}, Recall: {val_recall:.4f}, [test]: Acc: {test_acc:.4f}, AUPRC: {test_auprc:.4f} AUC: {test_auc:.4f}, Prec: {test_prec:.4f}, Recall: {test_recall:.4f}')

                    torch.save({
                        'epoch': epoch,
                        'val_res': {'acc': val_acc, 'auprc': val_auprc, 'auc': val_auc, 'prec': val_prec, 'recall': val_recall},
                        'test_res': {'acc': test_acc, 'auprc': test_auprc, 'auc': test_auc, 'prec': test_prec, 'recall': test_recall},
                        'encoder_state_dict': self.eeg_encoder.state_dict()
                    }, self.best_model_path)
            
        if self.is_ddp:
            dist.barrier()

    def train_one_epoch_binaryclass(self, data_loader, epoch=None, train=True):
        self.eeg_encoder.train() if train else self.eeg_encoder.eval()

        loss_log = []
        preds, targets, scores = [], [], []
        
        for step, (x, y) in enumerate(data_loader):
            # Update learning rate
            if (epoch is not None) and train:
                it = self.n_iters_per_epoch * epoch + step
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule_values[it]
                    
            # Forward pass
            x = x.float().to(self.args.device)
            y = y.to(self.args.device)
            pred = self.eeg_encoder(x, self.ch_idx).squeeze(-1)
            loss = self.criterion(pred, y.float())

            # Training step
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation step
            else:
                pred = pred.detach()
                score = torch.sigmoid(pred)  # Convert logits to probabilities
                pred_binary = (score > 0.5).float()  # Threshold at 0.5 for binary classification
                scores += score.cpu().numpy().tolist()
                preds += pred_binary.cpu().squeeze().numpy().tolist()
                targets += y.cpu().squeeze().numpy().tolist()
            
            loss_log.append(loss.data.cpu().numpy())
        
        if train:
            return np.mean(loss_log)
            
        # Calculate metrics
        targets = np.array(targets)
        preds = np.array(preds)
        acc, auprc, auc, prec, recall, cm = self.evaluator.get_binaryclass_metrics(targets, preds, scores)
        
        return np.mean(loss_log), acc, auprc, auc, prec, recall, cm
    
    def evaluate(self):
        # Load best model
        ckpt = torch.load(self.best_model_path, weights_only=False)
        self.eeg_encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.eeg_encoder.eval()

        if self.criterion is None:
            self._get_criterion()
        
        with torch.no_grad():
            # Test phase
            if self.args.num_class > 2:
                loss, acc, f1, kappa, cm = self.train_one_epoch_multiclass(self.test_loader, train=False)
                if self.is_ddp:
                    loss, acc, f1, kappa = ddp_reduce([loss, acc, f1, kappa], self.world_size, self.device)
                    cm = ddp_reduce(cm, 1, self.device)
                res_str = f'[Test] Loss: {loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}'
            elif self.args.num_class == 0:
                loss, corr, r2, rmse = self.train_one_epoch_regression(self.test_loader, train=False)
                if self.is_ddp:
                    loss, corr, r2, rmse = ddp_reduce([loss, corr, r2, rmse], self.world_size, self.device)
                res_str = f'[Test] Loss: {loss:.4f}, Corr: {corr:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}'
            else:
                loss, acc, auprc, auc, prec, recall, cm = self.train_one_epoch_binaryclass(self.test_loader, train=False)
                if self.is_ddp:
                    loss, acc, auprc, auc, prec, recall = ddp_reduce([loss, acc, auprc, auc, prec, recall], self.world_size, self.device)
                    cm = ddp_reduce(cm, 1, self.device)
                res_str = f'[Test] Loss: {loss:.4f}, Acc: {acc:.4f}, AUPRC: {auprc:.4f}, AUC: {auc:.4f}, Prec: {prec:.4f}, Recall: {recall:.4f}'

        if self.rank == 0:
            print(res_str)
            if self.args.num_class != 0:
                print('Confusion Matrix:')
                print(cm)

            # Save test results
            res_path = os.path.join(self.args.res_path, f'result.txt')
            with open(res_path, 'a') as f:
                if self.args.comment is not None:
                    f.write(f'#{self.args.comment}\n')
                f.write(res_str + '\n')
                if self.args.num_class != 0:
                    f.write('Confusion Matrix:\n')
                    f.write(str(cm) + '\n')

            # if wandb.run is not None:
            #     res_dict = {'test': res_str}
            #     if self.args.num_class != 0:
            #         res_dict['confusion matrix'] = wandb.Table(data=cm.tolist())
            #     wandb.log(res_dict)
