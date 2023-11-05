import os
import torch
import numpy as np
import torch.nn as nn


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        args.device = self.device
        self.model = self._build_model().to(self.device)
        self._select_criterion()
        self._select_optimizer()

    def _build_model(self):
        raise NotImplementedError
    
    def _get_data(self):
        raise NotImplementedError
    
    def _select_optimizer(self):
        raise NotImplementedError

    def _select_criterion(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def train(self):
        raise NotImplementedError

    def vali(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError