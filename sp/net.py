import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import *

import sp

###############################################################################
#
# Split Network
#
###############################################################################

class SpNet(nn.Module):
    def __init__(self):
        super(SpNet, self).__init__()
        self.net = None
        self.next_layers = {}
        self.previous_layers = {}
        self.layers_to_split = []
        self.verbose = True
        self.n_elites = 0

        self.num_group = 1
        
    def create_optimizer(self):
        pass

    def forward(self, x):
        pass

    def split(self):
        pass

    def clear(self):
        for layer in self.net:
            if isinstance(layer, sp.SpModule):
                layer.clear()

    def get_num_elites(self):
        n = 0
        for i in self.layers_to_split:
            n += self.net[i].module.weight.shape[0]
        self.n_elites = int(n * self.grow_ratio)

    def get_num_elites_group(self, group_num):
        for g in range(group_num):
            n = 0
            for i in self.layers_to_split_group[g]:
                n += self.net[i].module.weight.shape[0]
            try:
                self.n_elites_group[g] = int(n * self.grow_ratio)
            except:
                self.n_elites_group = {}
                self.n_elites_group[g] = int(n * self.grow_ratio)

    def sp_threshold(self):
        ws, wi = torch.sort(torch.cat([self.net[i].w for i in self.layers_to_split]).reshape(-1))
        total= ws.shape[0]
        threshold = ws[self.n_elites]
        return threshold

    def sp_threshold_group(self, group_num):
        ws, wi = torch.sort(torch.cat([self.net[i].w for i in self.layers_to_split_group[group_num]]).reshape(-1))
        total= ws.shape[0]
        threshold = ws[self.n_elites_group[group_num]]
        return threshold

    def save(self, path='./tmp.pt'):
        torch.save(self.state_dict(), path)

    def load(self, path='./tmp.pt'):
        self.load_state_dict(torch.load(path))

    def get_num_params(self):
        model_n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return model_n_params

    def spe(self, dataloader, n_batches):
        pass

    def spf(self, dataloader, n_batches):
        pass

    def spff(self, dataloader, n_batches):
        pass

    def split(self, split_method, dataset, n_batches=-1):
        self.num_group = 1 if self.config.model != 'mobile' else 2
        if split_method not in ['random', 'exact', 'fast', 'firefly', 'fireflyn']:
            raise NotImplementedError

        if self.verbose:
            print('[INFO] start splitting ...')

        start_time = time.time()
        self.net.eval()

        if self.num_group == 1:
            self.get_num_elites()
        else:
            self.get_num_elites_group(self.num_group)
        split_fn = {
            'exact': self.spe,
            'fast': self.spf,
            'firefly': self.spff,
            'fireflyn': self.spffn,
        }

        if split_method != 'random':
            split_fn[split_method](dataset, n_batches)

        n_neurons_added = {}

        if split_method == 'random':
            n_layers = len(self.layers_to_split)
            n_total_neurons = 0
            threshold = 0.
            for l in self.layers_to_split:
                n_total_neurons += self.net[l].get_n_neurons()
            n_grow = int(n_total_neurons * self.grow_ratio)
            n_new1 = np.random.choice(n_grow, n_layers, replace=False)
            n_new1 = np.sort(n_new1)
            n_news = []
            for i in range(len(n_new1) - 1):
                if i == 0:
                    n_news.append(n_new1[i])
                    n_news.append(n_new1[i + 1] - n_new1[i])
                else:
                    n_news.append(n_new1[i + 1] - n_new1[i])
                    n_news[-1] += 1
            for i, n_new_ in zip(reversed(self.layers_to_split), n_news):
                if isinstance(self.net[i], sp.SpModule) and self.net[i].can_split:
                    n_new, idx = self.net[i].random_split(n_new_)
                    n_neurons_added[i] = n_new
                    if n_new > 0: # we have indeed splitted this layer
                        for j in self.next_layers[i]:
                            self.net[j].passive_split(idx)
        elif split_method == 'fireflyn':
            if self.num_group == 1:
                threshold = self.sp_threshold()
            for i in reversed(self.layers_to_split):
                if isinstance(self.net[i], sp.SpModule) and self.net[i].can_split:
                    if self.num_group != 1:
                        group = self.total_group[i]
                        threshold = self.sp_threshold_group(group)
                    n_new, split_idx, new_idx = self.net[i].spffn_active_grow(threshold)
                    sp_new = split_idx.shape[0] if split_idx is not None else 0
                    n_neurons_added[i] = (sp_new, n_new-sp_new)
                    if self.net[i].kh == 1:
                        isfirst = True
                    else:
                        isfirst = False
                    for j in self.next_layers[i]:
                        print('passive', self.net[j].module.weight.shape)
                        self.net[j].spffn_passive_grow(split_idx, new_idx)

        else:
            threshold= self.sp_threshold()
            # actual splitting
            for i in reversed(self.layers_to_split):
                if isinstance(self.net[i], sp.SpModule) and self.net[i].can_split:
                    n_new, idx = self.net[i].active_split(threshold)
                    n_neurons_added[i] = n_new
                    if n_new > 0: # we have indeed splitted this layer
                        for j in self.next_layers[i]:
                            self.net[j].passive_split(idx)

        self.net.train()
        self.clear() # cleanup auxiliaries
        self.create_optimizer() # re-initialize optimizer

        end_time = time.time()
        if self.verbose:
            print('[INFO] splitting takes %10.4f sec. Threshold value is %10.9f' % (
                end_time - start_time, threshold))
            if split_method == 'fireflyn':
                print('[INFO] number of added neurons: \n%s\n' % \
                        '\n'.join(['-- %d grows (sp %d | new %d)' % (x, y1, y2) for x, (y1, y2) in n_neurons_added.items()]))
            else:
                print('[INFO] number of added neurons: \n%s\n' % \
                        '\n'.join(['-- %d grows %d neurons' % (x, y) for x, y in n_neurons_added.items()]))
        return n_neurons_added
