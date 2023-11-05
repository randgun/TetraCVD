from exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from tetra.Tetra_module import Tetra
from utils.tools import EarlyStopping, find_threshold_micro
from utils.metrics import all_metrics, print_metrics
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from transformers import AdamW, optimization
from torch.optim import lr_scheduler

import os
import time
import logging

import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class Exp_Tetra(Exp_Basic):
    def __init__(self, args):
        super(Exp_Tetra, self).__init__(args)
        self.alpha = torch.tensor(0.1).to(self.device)

    def _build_model(self):
        model = Tetra(self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, 'Multi_data')
        return data_set, data_loader

    def _select_scheduler(self):
        scheduler_map = {
            "get_linear_schedule_with_warmup": optimization.get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(self.args.warmup_rate * self.training_size) if self.args.warmup_step == -1 else self.args.warmup_step, 
                num_training_steps=self.training_size),

            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.train_epochs, 
                eta_min=self.args.min_lr),

            "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.train_epochs, 
                T_mult=1, 
                eta_min=self.args.min_lr),

            "get_cosine_schedule_with_warmup": optimization.get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=int(self.args.warmup_rate * self.training_size) if self.args.warmup_step == -1 else self.args.warmup_step, 
                num_training_steps=self.training_size,
                num_cycles=2,
                last_epoch=-1),
            
            "MultiStepLR": lr_scheduler.MultiStepLR(
                self.optimizer, 
                [self.args.train_epochs//5, self.args.train_epochs//2], 
                gamma=0.1),
        }
        self.scheduler = scheduler_map[self.args.scheduler]    
        
    def _select_optimizer(self):
        optimizer_grouped_parameters = self._get_optimizer_grouped_parameters()
        self.optimizer = AdamW(optimizer_grouped_parameters)

    def _select_criterion(self):
        if self.args.task == 'multi-class':
            self.criterion = nn.CrossEntropyLoss()
        else:
            # 多标签分类loss
            self.criterion = nn.BCEWithLogitsLoss()
    
    def _get_optimizer_grouped_parameters(self):
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in no_decay
                ],
                "weight_decay": 0,
                "lr": self.args.learning_rate[1],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in no_decay
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate[1],
            },
        ]
        return optimizer_grouped_parameters

    def train(self, setting):
        pre_epoch = 1
        train_data, train_loader = self._get_data(flag='train')
        self.training_size = (self.args.train_epochs - pre_epoch) * len(train_loader)
        print(self.training_size)
        self._select_scheduler()

        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
    
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.model.train()

        if self.args.tetra.mode == 'part':
            self.model.text_model.eval()
            for p in self.model.text_model.parameters():
                p.requires_grad = False

            self.model.ts_model.eval()
            for p in self.model.ts_model.parameters():
                p.requires_grad = False
        else:
            self.model.text_model.backbone.eval()
            for p in self.model.text_model.backbone.parameters():
                p.requires_grad = False
            
            self.model.ts_model.eval()
            for p in self.model.ts_model.parameters():
                p.requires_grad = False

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate[2], momentum=0.9)

        for epoch in range(self.args.train_epochs):

            if epoch == pre_epoch:
                self.model.text_model.train()
                for p in self.model.text_model.parameters():
                    p.requires_grad = True

                self.model.text_model.backbone.eval()
                for p in self.model.text_model.backbone.parameters():
                    p.requires_grad = False

                self.model.ts_model.train()
                for p in self.model.ts_model.parameters():
                    p.requires_grad = True
                
                self._select_optimizer()

            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            
            for i, batch in enumerate(train_loader):

                iter_count += 1
                self.optimizer.zero_grad()

                text_ids = batch["input_ids"].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_mask = batch['seq_mask'].to(self.device)
                ts_P = batch["P"].to(self.device)
                ts_Ptime = batch["Ptime"].to(self.device)
                ts_Pstatic = batch["Pstatic"].to(self.device)
                y = batch['labels'].to(self.device)

                ts_P = ts_P.permute(1, 0, 2)
                ts_Ptime = ts_Ptime.permute(1, 0)

                # 有效时间长度
                lengths = torch.sum(ts_Ptime > 0, dim=0)
                # print(y)

                inputs = [text_ids, attention_mask, seq_mask, ts_P, ts_Ptime, ts_Pstatic, lengths]
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.forward(inputs)
                        loss = self.criterion(outputs["probs"], y)
                        loss += self.alpha * outputs['itc_loss']
                        train_loss.append(loss.item())
                else:
                    outputs = self.model.forward(inputs)
                    loss = self.criterion(outputs["probs"], y)
                    loss += self.alpha * outputs['itc_loss']
                    train_loss.append(loss.item())

                # TODO 修改打印loss频率
                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.scheduler.step()

            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if not self.args.train_only:
                metric = self.vali(vali_loader)
                if self.args.task == 'multi-class':
                    loss_val, acc_val, aupr_val, auc_val = metric[0:4]
                
                    print("Epoch: {0}, Loss_val: {1:.4f}, Acc_val: {2:.4f}, Aupr_val: {3:.4f}, Auc_val: {4:.4f}".format
                            (epoch + 1, loss_val, acc_val, aupr_val, auc_val))
                    early_stopping(loss_val, self.model, path)
                else:
                    print_metrics(metric)
                    early_stopping(metric['loss'], self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


    def eval(self, data_loader, tqdm_bar=None):
        self.model.eval()
        outputs = []
        it = tqdm(data_loader) if tqdm_bar else data_loader
        with torch.no_grad():
            for batch in it:
                text_ids = batch["input_ids"].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_mask = batch['seq_mask'].to(self.device)
                ts_P = batch["P"].to(self.device)
                ts_Ptime = batch["Ptime"].to(self.device)
                ts_Pstatic = batch["Pstatic"].to(self.device)
                y = batch['labels']
                
                ts_P = ts_P.permute(1, 0, 2)
                ts_Ptime = ts_Ptime.permute(1, 0)

                # 有效时间长度
                lengths = torch.sum(ts_Ptime > 0, dim=0)
                inputs = [text_ids, attention_mask, seq_mask, ts_P, ts_Ptime, ts_Pstatic, lengths]

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        res = self.model.forward(inputs)
                else:
                    res = self.model.forward(inputs)

                yhat_raw = res['probs'].cpu()
                loss = self.criterion(yhat_raw.cpu(), y)
                loss += (self.alpha * res['itc_loss']).cpu()
                res = {'y': y, 'yhat_raw': yhat_raw, 'loss': loss.reshape(1)}
                outputs.append({key:value.cpu().detach() for key, value in res.items()})
        
        y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
        yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
        loss = torch.cat([output['loss'] for output in outputs]).cpu().detach().numpy()
        self.model.train()
        return y, yhat_raw, loss
    
    def vali(self, data_loader, threshold=None, tqdm_bar=None):
        y, yhat_raw, loss = self.eval(data_loader, tqdm_bar)
        total_loss = np.average(loss)
        if self.args.task == 'multi-class':
            acc_val = np.sum(y.ravel() == yhat_raw.ravel()) / yhat_raw.shape[0]
            auc_val = roc_auc_score(y, yhat_raw)
            aupr_val = average_precision_score(y, yhat_raw)
            return (total_loss, acc_val, auc_val, aupr_val, y, yhat_raw)
        
        elif self.args.task == 'multi-label':
            if threshold is None:
                threshold = find_threshold_micro(yhat_raw, y)
            yhat = np.where(yhat_raw > threshold, 1, 0)
            metric = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
            metric['loss'] = total_loss
            metric['y'] = y
            metric['yhat'] = yhat
            return metric
        else:
            raise NotImplementedError
    
    def test(self, setting, test=0, threshold=None, tqdm_bar=True):
        _, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        metric = self.vali(test_loader, threshold=threshold, tqdm_bar=tqdm_bar)
        return metric

    