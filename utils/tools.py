import logging
import time
import os
import torch
import numpy as np
import pandas as pd
import random
import ast
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F

"""
@brief
Some tricks for trianing better deep learning model
Reference: https://github.com/TelmaZzzz/Trainer/blob/master/src/Utils.py
"""
def set_logger(logger=None, filename='train'):
    # 分别将日志写到输出流和指定文件
    if logger == None:
        raise TypeError("A logger object should be init first")
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', \
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    
    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.WARNING)
    handler1.setFormatter(formatter)

    handler2 = logging.FileHandler(filename=f"{filename}.log")
    handler2.setLevel(logging.DEBUG)
    handler2.setFormatter(formatter)
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    
    return logger

def timer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info("Function \"{}\" run {:.2f}s.".format(func.__name__, end_time - start_time))
        return res

    return deco

def seed_everything(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
'''
Tricks
EMA: 权重的更新采用指数移动平均的方式
SWA: 在优化的末期取k个优化轨迹上的checkpoints, 平均他们的权重, 得到最终的网络权重
## 对抗训练
FGM: Fast Gradient Method
PGD: Projected Gradient Descent

'''
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.6, emb_name='embeddings.'):
        # emb_name这个参数要换成模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, emb_name="embeddings.", epsilon=0.6, alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        device=None,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.device = device
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            _, adv_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            self.optimizer.zero_grad()
            adv_loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def df_analyze(df:pd.DataFrame):
    assert len(df.index) != 0
    null_series = df.isnull().sum()
    unique_series = df.nunique()
    type_series = df.dtypes
    
    result = pd.DataFrame({'null':null_series, 'unique':unique_series, 'type':type_series})
    result.reset_index(drop=False, inplace=True)
    result.rename(columns={'index':'Column'}, inplace=True)
    # print(result)
    df_shape = df.shape
    # print('This DataFrame with {} rows x {} columns'.format(df_shape[0], df_shape[1]))
    return result


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
    
class YamlObject:
    '''
    yaml文件对象化
    '''
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                setattr(self, key, YamlObject(**value))
            else:
                setattr(self, key, value)
        
        
def randomSplit(n, save_path, K_splits = 5, p_train=0.8, p_val=0.1):
    """Use 8:1:1 split"""
    # n is total number of samples

    n_train = round(n*p_train)
    n_val = round(n*p_val)
    n_test = n - (n_train+n_val)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for j in range(K_splits):
        p = np.random.permutation(n)
        idx_train = p[:n_train]
        idx_val = p[n_train:n_train+n_val]
        idx_test = p[n_train+n_val:]
        np.save(save_path + str(j+1) + '_fold.npy', (idx_train, idx_val, idx_test))


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

    
def itc_loss(emb_i, emb_j, temperature):	
    # reference: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision 2021

    z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
    z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
    batch_size = emb_i.shape[0]


    representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2) / temperature 
    
    sim_ij = torch.diag(similarity_matrix, batch_size)      # bs
    sim_ji = torch.diag(similarity_matrix, -batch_size)     # bs
    
    # right-top part
    mat_1 = similarity_matrix[:batch_size, batch_size:]
    # left-bottom part
    mat_2 = similarity_matrix[batch_size:, :batch_size]

    mat_1 = torch.exp(mat_1)
    mat_2 = torch.exp(mat_2)
    sim_ij = torch.exp(sim_ij)
    sim_ji = torch.exp(sim_ji)

    denominator_1 = torch.sum(mat_1, dim=-1)
    denominator_2 = torch.sum(mat_2, dim=-1)

    loss_1 = -torch.sum(torch.log(sim_ij / denominator_1)) / batch_size
    loss_2 = -torch.sum(torch.log(sim_ji / denominator_2)) / batch_size

    return loss_1 + loss_2


def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    # print(threshold)
    return threshold


    
    