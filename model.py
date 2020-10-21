from torch.utils.data import DataLoader, ConcatDataset
import copy
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import *

import sys
import sp

def sp_vgg(model, n_classes=10, dimh=16, method='none'):
    cfgs = {
        'vgg11': [1, 'M', 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
        'vgg14': [1, 1, 'M', 2, 2, 'M', 4, 4, 'M', 8, 8, 'M', 8, 8, 'M'],
        'vgg16': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 'M', 8, 8, 8, 'M', 8, 8, 8, 'M'],
        'vgg19': [1, 1, 'M', 2, 2, 'M', 4, 4, 4, 4, 'M', 8, 8, 8, 8, 'M', 8, 8, 8, 8, 'M'],
    }
    cfg = cfgs[model]
    next_layers = {}
    prev_idx = -1
    in_channels = 3
    net = nn.ModuleList()
    n = len(cfg)
    for i, x in enumerate(cfg):
        if x == 'M':
            net.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif x == 'A':
            net.append(nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            if method == 'none':
                net.append(sp.Conv2d(in_channels, 64*x, kernel_size=3, padding=1, actv_fn='relu', has_bn=True))
                in_channels = 64*x
            else:
                net.append(sp.Conv2d(in_channels, dimh, kernel_size=3, padding=1, actv_fn='relu', has_bn=True))
                in_channels = dimh
            if prev_idx >= 0: next_layers[prev_idx] = [i]
            prev_idx = i
    net.append(sp.Conv2d(in_channels, n_classes, kernel_size=1, padding=0, actv_fn='none', can_split=False))
    next_layers[prev_idx] = [n]
    layer2split = list(next_layers.keys())
    return net, next_layers, layer2split

def sp_mobile(n_classes=10, dimh=16, method='none'):
    cfg = [32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    if method != 'none':
        cfg = [32, 32, (32, 2), 32, (32, 2), 32, (32, 2), 32, 32, 32, 32, 32, (32, 2), 32]
    net = nn.ModuleList()
    net.append(sp.Conv2d(3, cfg[0], 3, padding=1, actv_fn='swish', has_bn=True))

    next_layers = {}
    for i in range(26):
        next_layers[i] = [i+1, i+2]

    in_c = cfg[0]
    for out_c in cfg[1:]:
        if isinstance(out_c, int):
            out_c, s = out_c, 1
        else:
            out_c, s = out_c
        net.append(sp.Conv2d(in_c,
                             in_c,
                             kernel_size=3,
                             stride=s,
                             padding=1,
                             groups=in_c,
                             actv_fn='swish',
                             has_bn=True,
                             can_split=False))
        net.append(sp.Conv2d(in_c,
                             out_c,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             actv_fn='swish',
                             has_bn=True))
        in_c = out_c
    net.append(nn.AvgPool2d(2))
    net.append(sp.Conv2d(in_c, n_classes, kernel_size=1, padding=0, actv_fn='none', can_split=False))
    next_layers[26] = [28]
    layer2split = [0,2,4,6,8,10,12,14,16,18,20,22,24,26]
    layer2split_group = {0: [0,2,4,6], 1: [8,10,12,14,16,18,20,22, 24, 26]}
    return net, next_layers, layer2split, layer2split_group


class Classifier(sp.SpNet):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.verbose = config.verbose
        self.device = config.device
        self.grow_ratio = config.grow_ratio

        C, H, W = config.dim_input
        assert (H == 32) and (W == 32)
        dimh = config.dim_hidden


        if 'mobile' in config.model:
            self.net, self.next_layers, self.layers_to_split, self.layers_to_split_group = sp_mobile(n_classes = config.dim_output, dimh=dimh, method=self.config.method)
            self.total_group = {}
            for i in range(2):
                group = self.layers_to_split_group[i]
                for j in group:
                    self.total_group[j] = i
        else:
            self.net, self.next_layers, self.layers_to_split = sp_vgg(config.model, n_classes = config.dim_output,  dimh=dimh, method=self.config.method)

        if self.verbose:
            print("[INFO] network architecture: ", self.next_layers)

        self.lr = 0.1
        self.grow_ratio = config.grow_ratio
        self.create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def create_optimizer(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)

        if self.config.optimizer == 'Adam':
            self.opt = Adam(nn.ParameterList(params),
                            lr=self.lr,
                            betas=(self.config.beta1, self.config.beta2),
                            weight_decay=self.config.weight_decay)

        elif self.config.optimizer == 'SGD':
            self.opt = SGD(nn.ParameterList(params),
                           lr=self.lr,
                           momentum=self.config.momentum,
                           weight_decay=self.config.weight_decay,
                           nesterov=True)

    def get_cfg(self):
        cfg = []
        for layer in self.net:
            if isinstance(layer, sp.Conv2d):
                cfg.append(layer.module.weight.shape[1])
            elif isinstance(layer, nn.MaxPool2d):
                cfg.append('M')
            elif isinstance(layer, nn.AvgPool2d):
                cfg.append('A')
        return cfg

    def set_lr(self, lr):
        self.lr = lr
    def decay_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] *= 0.1

    ## -- forward -- ##
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        return x

    def spffn_forward(self, x, alpha):
        for layer in self.net:
            #if isinstance(layer, sp.SpModule) and layer.can_split:
            if isinstance(layer, sp.SpModule):
                x = layer.spffn_forward(x, alpha=alpha)
            else:
                x = layer(x)
        return x.view(x.shape[0], -1)

    def spff_forward(self, x, alpha):
        for layer in self.net:
            if isinstance(layer, sp.SpModule) and layer.can_split:
                x = layer.spff_forward(x, alpha=alpha)
            else:
                x = layer(x)
        return x.view(x.shape[0], -1)

    def spf_forward(self, x):
        for layer in self.net:
            if isinstance(layer, sp.SpModule) and layer.can_split:
                x = layer.spf_forward(x)
            else:
                x = layer(x)
        return x.view(x.shape[0], -1)

    def spe_forward(self, x, split_layer):
        for i, layer in enumerate(self.net):
            if isinstance(layer, sp.SpModule) and layer.can_split and split_layer == i:
                x = layer.spe_forward(x)
            else:
                x = layer(x)
        return x.view(x.shape[0], -1)

    def classify(self, x):
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                x = layer(x)
            x = x.view(x.shape[0], -1)
        return x.argmax(-1)

    ## -- loss function -- ##
    def loss_fn(self, x, y):
        scores = self.forward(x)
        loss = self.criterion(scores, y)
        return loss

    def spffn_loss_fn(self, x, y, alpha=-1):
        scores = self.spffn_forward(x, alpha=alpha)
        loss = self.criterion(scores, y)
        return loss

    def spff_loss_fn(self, x, y, alpha=-1):
        scores = self.spff_forward(x, alpha=alpha)
        loss = self.criterion(scores, y)
        return loss

    def spf_loss_fn(self, x, y):
        scores = self.spf_forward(x)
        loss = self.criterion(scores, y)
        return loss

    def spe_loss_fn(self, x, y, split_layer=-1):
        scores = self.spe_forward(x, split_layer=split_layer)
        loss = self.criterion(scores, y)
        return loss

    ## -- update -- ##
    def update(self, x, y):
        loss = self.loss_fn(x, y) # for current task
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().item()

    ## -- firefly new split -- ##
    def spffn(self, dataset, n_batches):
        v_params = []
        for i, layer in enumerate(self.net):
            if isinstance(layer, sp.SpModule):
                enlarge_in = (i > 0)
                enlarge_out = (i < len(self.net)-1)
                self.net[i].spffn_add_new(enlarge_in=enlarge_in, enlarge_out=enlarge_out)
                self.net[i].spffn_reset()
                if layer.can_split:
                    v_params += [self.net[i].v]
                if enlarge_in:
                    v_params += [self.net[i].vni]
                if enlarge_out:
                    v_params += [self.net[i].vno]

        opt_v = RMSprop(nn.ParameterList(v_params), lr=1e-3, momentum=0.1, alpha=0.9)

        # train the splitting direction
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        n_batches = len(loader)

        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            loss = self.spffn_loss_fn(x, y)
            opt_v.zero_grad()
            loss.backward()
            for layer in self.net:
                if isinstance(layer, sp.SpModule):
                    layer.spffn_penalty()
            opt_v.step()

        self.config.granularity = 1
        alphas = np.linspace(0, 1, self.config.granularity*2+1)
        for alpha in alphas[1::2]:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.spffn_loss_fn(x, y, alpha=1.0)
                opt_v.zero_grad()
                loss.backward()
                for i in self.layers_to_split:
                    self.net[i].spffn_update_w(self.config.granularity * n_batches, output = False)


    ## -- firefly split -- ##
    def spff(self, dataset, n_batches):
        v_params = []
        for i in self.layers_to_split:
            if self.net[i].can_split:
                self.net[i].spff_reset()
                v_params += [self.net[i].v]

        opt_v = RMSprop(nn.ParameterList(v_params), lr=1e-3, momentum=0.1, alpha=0.9)

        # train the splitting direction
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        n_batches = len(loader)

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = self.spff_loss_fn(x, y)
            opt_v.zero_grad()
            loss.backward()
            for i in self.layers_to_split:
                loss += self.net[i].v.norm(1) * 1e-2
            opt_v.step()

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = self.spff_loss_fn(x, y, alpha=1e-3)
            opt_v.zero_grad()
            loss.backward()
            for i in self.layers_to_split:
                self.net[i].spff_update_w(n_batches)

        for i in self.layers_to_split:
            self.net[i].spff_scale_v()

    ## -- fast split (Rayleigh Quotient) -- ##
    def spf(self, dataset, n_batches):
        params = []
        for i in self.layers_to_split:
            self.net[i].spf_reset()
            params += [self.net[i].v]

        opt_v = RMSprop(params, lr=1e-3, momentum=0.1, alpha=0.9)

        # train the splitting direction
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        n_batches = len(loader)
        for e in range(10):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.spf_loss_fn(x, y)
                opt_v.zero_grad()
                loss.backward()
                for i in self.layers_to_split:
                    self.net[i].spf_update_v()
                opt_v.step()

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            loss = self.spf_loss_fn(x, y)
            opt_v.zero_grad()
            loss.backward()
            for i in self.layers_to_split:
                self.net[i].spf_update_w(n_batches)

    ## -- exact split -- ##
    def spe(self, dataset, n_batches):
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        n_batches = len(loader) // 10
        for i in self.layers_to_split:
            self.opt.zero_grad()
            io = 0
            for x, y in loader:
                io += 1
                x, y = x.to(self.device), y.to(self.device)
                loss = self.spe_loss_fn(x, y, split_layer=i)
                loss.backward()
                if io >= n_batches:
                    break
            self.net[i].spe_eigen(float(n_batches)/10)
