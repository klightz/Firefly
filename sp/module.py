import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import *

###############################################################################
#
# Split Block Abstract Class
#
###############################################################################

class SpModule(nn.Module):
    def __init__(self,
                 can_split=True,
                 actv_fn='relu',
                 has_bn=False,
                 has_bias=True,
                 rescale=1.0):
        super(SpModule, self).__init__()

        # properties
        self.can_split = can_split
        self.actv_fn = actv_fn
        self.has_bn = has_bn
        self.has_bias = has_bias
        self.epsilon = 1e-2
        self.K = 70

        # modules
        self.module = None
        self.bn = None

        # auxiliaries
        self.w = None
        self.v = None
        self.y = None
        self.S = []

        self.leaky_alpha = 0.2

    def clear(self):
        del self.w
        del self.v
        del self.y
        del self.S
        try:
            del self.vni
        except:
            pass
        try:
            del self.vno
        except:
            pass
        self.w = self.v = self.y = None
        self.S = []

    def get_device(self):
        try:
            return 'cuda' if self.module.weight.data.is_cuda else 'cpu'
        except:
            raise Exception('[ERROR] no module initialized')

    def _d2_actv(self, x, beta=3.):
        if self.actv_fn == 'relu':
            # use 2nd order derivative of softplus for approximation
            s = torch.sigmoid(x*beta)
            return beta*s*(1.-s)
        elif self.actv_fn == 'softplus':
            s = torch.sigmoid(x)
            return s*(1.-s)
        elif self.actv_fn == 'rbf':
            return (x.pow(2)-1)*(-x.pow(2)/2).exp()
        elif self.actv_fn == 'leaky_relu':
            s = torch.sigmoid(x*beta)
            return beta*s*(1.-s)*(1.-self.leaky_alpha)
        elif self.actv_fn == 'swish':
            s = torch.sigmoid(x)
            return s*(1.-s) + s + x*s*(1.-s) - (s.pow(2) + 2.*x*s.pow(2)*(1.-s))
        elif self.actv_fn == 'sigmoid':
            s = torch.sigmoid(x)
            return (s-s.pow(2)) * (1.-s).pow(2)
        elif self.actv_fn == 'tanh':
            h = torch.tanh(x)
            return -2.*h * (1-h.pow(2))
        elif self.actv_fn == 'none':
            return torch.ones_like(x)
        else:
            raise NotImplementedError

    def _activate(self, x):
        if self.actv_fn == 'relu':
            return F.relu(x)
        elif self.actv_fn == 'leaky_relu':
            return F.leaky_relu(x, self.leaky_alpha)
        elif self.actv_fn == 'swish':
            return x * torch.sigmoid(x)
        elif self.actv_fn == 'rbf':
            return (-x.pow(2)/2).exp()
        elif self.actv_fn == 'sigmoid':
            return torch.sigmoid(x)
        elif self.actv_fn == 'tanh':
            return torch.tanh(x)
        elif self.actv_fn == 'softplus':
            return F.softplus(x)
        elif self.actv_fn == 'none':
            return x
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.module(x)
        if self.has_bn:
            x = self.bn(x)
        return self._activate(x)

    def active_split(self, threshold):
        pass

    def passive_split(self, idx):
        pass
