import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, grad
from torch.distributions import Normal
from torch.optim import *
from .module import SpModule

###############################################################################
#
# Conv2d Split Layer
#
###############################################################################

class Conv2d(SpModule):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            groups = 1,
            can_split=True,
            bias=True,
            actv_fn='relu',
            has_bn=False,
            rescale=1.0):

        super().__init__(can_split=can_split,
                         actv_fn=actv_fn,
                         has_bn=has_bn,
                         has_bias=bias,
                         rescale=rescale)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
            self.has_bias = False

        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kh, self.kw = kernel_size

        if isinstance(padding, int):
            self.ph = self.pw = padding
        else:
            assert len(padding) == 2
            self.ph, self.pw = padding

        if isinstance(stride, int):
            self.dh = self.dw = stride
        else:
            assert len(stride) == 2
            self.dh, self.dw = stride
        self.groups = groups
        self.module = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                groups = groups,
                                stride=stride,
                                padding=padding,
                                bias=self.has_bias)


    def get_conv_patches(self, x):
        x = F.pad(x, (self.pw, self.pw, self.ph, self.ph))  # pad (left, right, top, bottom)
        # get all image windows of size (kh, kw) and stride (dh, dw)
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, H, W, C_in, kh, kw]
        return patches

    ###########################################################################
    # fast split
    ###########################################################################

    def spf_reset(self):
        # y is a dummy variable for storing gradients of v
        W = self.module.weight.data
        self.y = nn.Parameter(torch.zeros_like(W))
        self.y.retain_grad()
        self.v = nn.Parameter(torch.zeros_like(W))
        self.v.data.uniform_(-1e-1, 1e-1)
        self.v.retain_grad()
        self.w = 0.

    def spf_update_v(self):
        v = self.v
        sv = self.y.grad
        vv = v.pow(2).sum([1,2,3], keepdim=True)
        vsv = (sv * v).sum([1,2,3], keepdim=True)
        v_grad = 2. * (sv * vv - v * vsv) / vv.pow(2)
        self.v.grad = v_grad
        self.y.grad = None

    def spf_update_w(self, n=1.):
        v = self.v
        sv = self.y.grad
        vv = v.pow(2).sum([1,2,3])
        vsv = (sv * v).sum([1,2,3])
        self.w += (vsv / vv).data.clone() / n

    def spf_forward(self, x):
        out = self.module(x) # [B, C_out]
        bn_coef = 1.
        if self.has_bn:
            self.bn.eval() # fix running mean/variance
            out = self.bn(out)
            # calculate bn_coef
            bn_coef = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight
            bn_coef = bn_coef.view(1, -1, 1, 1) # [1, C_out, 1, 1]

        # normalize v
        v_norm = self.v.pow(2).sum([1,2,3], keepdim=True).sqrt().data
        self.v.data = self.v.data / v_norm

        patches = self.get_conv_patches(x)
        B, H, W, C_in, kh, kw = patches.size()

        x = patches.reshape(B*H*W, -1)

        left = x.mm(self.y.view(-1, C_in*kh*kw).t()).view(B, H, W, -1).permute(0,3,1,2)
        right = x.mm(self.v.view(-1, C_in*kh*kw).t()).view(B, H, W, -1).permute(0,3,1,2)

        aux = self._d2_actv(out) * (bn_coef*left) * (bn_coef*right)
        out = self._activate(out) + aux
        return out

    ###########################################################################
    # firefly split + new neurons
    ###########################################################################

    def spffn_add_new(self, enlarge_out=True, enlarge_in=True):
        self.eout = self.K if enlarge_out else 0
        self.ein = self.K if enlarge_in else 0
        if self.groups == 1:
            C_out, C_in = self.module.weight.data.shape[:2]
        else:
            C_out, _ = self.module.weight.data.shape[:2]
            C_in = C_out
        device = self.get_device()
        if self.has_bn and self.eout > 0:
            new_bn = nn.BatchNorm2d(C_out+self.eout).to(device)
            new_bn.weight.data[:C_out] = self.bn.weight.data.clone()
            new_bn.bias.data[:C_out] = self.bn.bias.data.clone()
            new_bn.running_mean.data[:C_out] = self.bn.running_mean.data.clone()
            new_bn.running_var.data[:C_out] = self.bn.running_var.data.clone()
            new_bn.weight.data[C_out:] = 1.
            new_bn.bias.data[C_out:] = 0.
            self.bn = new_bn
            self.bn.eval()
        if self.groups != 1:
            self.groups += self.K
        new_layer = nn.Conv2d(in_channels=C_in+self.ein,
                              out_channels=C_out+self.eout,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias, groups = self.groups).to(device)

        new_layer.weight.data[:C_out, :C_in, :, :] = self.module.weight.data.clone()

        if self.ein > 0:
            new_layer.weight.data[:, C_in:, :, :] = 0.
        if self.eout > 0:
            new_layer.weight.data[C_out:, :, :, :] = 0.
        self.module = new_layer
        self.module.eval()

    def spffn_penalty(self):
        penalty = 0.
        if self.can_split: penalty += self.v.pow(2).sum()
        if self.eout > 0: penalty += 1e-2 * self.vno.pow(2).sum()
        if penalty > 0: (penalty * 1e-2).backward()

    def spffn_clip(self):
        if self.ein > 0: # since output is just 1
            self.vni.data.clamp_(-1e-2, 1e-2)

    def spffn_reset(self):
        if self.groups == 1:
            C_out, C_in, kh, kw = self.module.weight.data.shape
        else:
            C_out, C_in, kh, kw = self.module.weight.data.shape
            C_in = C_out
        device = self.get_device()
        self.y = nn.Parameter(torch.zeros(1,C_out,1,1)).to(device)
        self.y.retain_grad()
        self.w = 0.

        if self.can_split:
            v = torch.zeros(C_out-self.eout, C_in-self.ein,kh,kw).to(device)
            v.uniform_(-1e-1, 1e-1)
            self.v = nn.Parameter(v)

        if self.ein > 0:
            vni = torch.zeros(C_out, self.ein, kh, kw).to(device)
            vni.uniform_(-1e-2, 1e-2)
            self.vni = nn.Parameter(vni)

        if self.eout > 0:
            vno = torch.zeros(self.eout, C_in-self.ein, kh, kw).to(device)
            n = kh * kw * (C_in - self.ein)
            stdv = 1. / math.sqrt(n)
            #vno.uniform_(-stdv, stdv)
            vno.normal_(0, 0.1)
            self.vno = nn.Parameter(vno)

    def spffn_update_w(self, d, output = False):
        if not output:
            self.w += (self.y.grad.data / d).view(-1)
            self.y.grad = None
        else:
            y_grad = grad(self.output.mean(), self.y)
            self.w +=  (self.y.grad.data / y_grad[0].data / d).view(-1)
            self.y.grad = None

    def spffn_forward(self, x, alpha=-1):
        out = self.module(x) # [out+eout, in+ein, H, W]

        patches = self.get_conv_patches(x)
        B, H, W, C_in, kh, kw = patches.size()
        C_out = out.shape[1]
        cin, cout = C_in - self.ein, C_out - self.eout

        x = patches.view(B*H*W, -1, kh*kw)

        if self.ein > 0:
            x1, x2 = x[:,:cin,:].view(B*H*W, -1), x[:,cin:,:].view(B*H*W,-1)
        else:
            x1 = x.view(B*H*W, -1)

        if self.can_split:
            noise_v = x1.mm(self.v.view(-1, cin*kh*kw).t()).view(B,H,W,-1).permute(0,3,1,2) # [B,cout,H,W]
            if alpha >= 0.:
                noise_v = (noise_v.detach() * self.y[:,:cout,:,:] + noise_v * alpha)

        if self.eout > 0:
            noise_vo = x1.mm(self.vno.view(-1, cin*kh*kw).t()).view(B,H,W,-1).permute(0,3,1,2)
            if alpha >= 0.:
                noise_vo = (noise_vo.detach() * self.y[:,cout:,:,:] + noise_vo * alpha)

        if self.ein > 0:
            noise_vi1 = x2.mm(self.vni.view(-1, self.ein*kh*kw).t())
            if self.eout > 0:
                noise_vi1, noise_vi2 = noise_vi1[:,:cout], noise_vi1[:,cout:] # [B*H*W, cout/eout]
                noise_vi1 = noise_vi1.view(B,H,W,-1).permute(0,3,1,2)
                noise_vi2 = noise_vi2.view(B,H,W,-1).permute(0,3,1,2)
            else:
                noise_vi1 = noise_vi1.view(B,H,W,-1).permute(0,3,1,2)

        o1_plus = o1_minus = o2 = 0.

        if self.can_split:
            o1_plus = out[:,:cout,:,:] + noise_v # [B, cout, H, W]
            o1_minus = out[:,:cout,:,:] - noise_v # [B, cout, H, W]
            if self.eout > 0:
                o2 = out[:,cout:,:,:] + noise_vo
            if self.ein > 0:
                o1_plus = o1_plus + noise_vi1
                o1_minus = o1_minus + noise_vi1
                if self.eout > 0:
                    o2 = o2 + noise_vi2
            if self.eout > 0:
                o1_plus = torch.cat((o1_plus, o2), 1)
                o1_minus = torch.cat((o1_minus, o2), 1)

            if self.has_bn:
                o1_plus = self.bn(o1_plus)
                o1_minus = self.bn(o1_minus)
            o1_plus = self._activate(o1_plus)
            o1_minus = self._activate(o1_minus)
            output = (o1_plus + o1_minus) / 2.
        else:
            o1 = out[:,:cout,:,:]
            if self.eout > 0:
                o2 = out[:,cout:,:,:] + noise_vo
                if self.ein > 0:
                    o2 = o2 + noise_vi2
            if self.ein > 0:
                o1 = o1 + noise_vi1
            if self.eout > 0:
                o1 = torch.cat((o1, o2), 1)
            if self.has_bn:
                o1 = self.bn(o1)
            output = self._activate(o1)
        self.output = output
        return output

    ###########################################################################
    # firefly split
    ###########################################################################

    def spff_reset(self):
        W = self.module.weight.data
        device = self.get_device()
        self.y = nn.Parameter(torch.zeros(1, W.shape[0], 1, 1)).to(device)
        self.y.retain_grad()
        self.v = nn.Parameter(torch.zeros_like(W))
        self.v.data.uniform_(-1e-1, 1e-1)
        self.w = 0.

    def spff_update_w(self, d):
        self.w += (self.y.grad.data/d).view(-1)

    def spff_scale_v(self):
        self.v.data = self.v.data * 1e2

    def spff_forward(self, x, alpha=-1):
        out = self.module(x)

        patches = self.get_conv_patches(x)
        B, H, W, C_in, kh, kw = patches.size()
        x = patches.view(B*H*W, -1)

        if alpha >= 0.:
            noise_out = x.mm(self.v.view(-1, C_in*kh*kw).t())
            noise_out = noise_out.view(B, H, W, -1).permute(0, 3, 1, 2)
            noise_out = (self.y * noise_out.detach() + noise_out * alpha)
        else:
            noise_out = x.mm(self.v.view(-1, C_in*kh*kw).t())
            noise_out = noise_out.view(B, H, W, -1).permute(0, 3, 1, 2)

        out_plus = out + noise_out
        out_minus = out - noise_out

        if self.has_bn:
            self.bn.eval()
            out_plus = self.bn(out_plus)
            out_minus = self.bn(out_minus)

        out_plus = self._activate(out_plus)
        out_minus = self._activate(out_minus)

        return (out_plus + out_minus) / 2.

    ###########################################################################
    # exact split
    ###########################################################################
    def spe_forward(self, x):
        out = self.module(x) # [B, C_out, H, W]

        if self.has_bn:
            self.bn.eval() # fix running mean/variance
            out = self.bn(out)
            # calculate bn_coff
            bn_coff = 1. / torch.sqrt(self.bn.running_var + 1e-5) * self.bn.weight
            bn_coff = bn_coff.view(1, -1, 1, 1) # [1, C_out, 1, 1]

        first_run = (len(self.S) == 0)

        # calculate 2nd order derivative of the activation
        nabla2_out = self._d2_actv(out) # [B, C_out, H, W]
        patches = self.get_conv_patches(x)
        B, H, W, C_in, KH, KW = patches.size()
        C_out = out.shape[1]

        D = C_in * KH * KW
        x = patches.view(B, H, W, D)

        device = self.get_device()
        auxs = [] # separate calculations for each neuron for space efficiency
        for neuron_idx in range(C_out):
            c = bn_coff[:, neuron_idx:neuron_idx+1, :, :] if self.has_bn else 1.
            l = c * x
            if first_run:
                S = Variable(torch.zeros(D, D).to(device), requires_grad=True) # [H_in, H_in]
                self.S.append(S)
            else:
                S = self.S[neuron_idx]
            aux = l.view(-1, D).mm(S).unsqueeze(1).bmm(l.view(-1, D, 1)).squeeze(-1) # (Bx)S(Bx^T), [B*H*W,1]
            aux = aux.view(B, 1, H, W)
            auxs.append(aux)

        auxs = torch.cat(auxs, 1) # [B, C_out, H, W]
        auxs = auxs * nabla2_out # [B, C_out, H, W]
        out = self._activate(out) + auxs
        return out

    def spe_eigen(self, avg_over=1.):
        A = np.array([item.grad.data.cpu().numpy() for item in self.S]) # [C_out, D, D]
        A /= avg_over
        A = (A + np.transpose(A, [0, 2, 1])) / 2
        w, v = np.linalg.eig(A) # [C_out, K], [C_out, D, K]
        w = np.real(w)
        v = np.real(v)
        min_idx = np.argmin(w, axis=1)
        w_min = np.min(w, axis=1) # [C_out,]
        v_min = v[np.arange(w_min.shape[0]), :, min_idx] # [C_out, D]
        self.w = w_min
        self.v = v_min
        device = self.get_device()
        self.w = torch.FloatTensor(w_min).to(device)
        self.v = torch.FloatTensor(v_min).to(device)
        self.v = self.v.view(*self.module.weight.data.shape)
        del A

    ## below are for copying weights and actual splitting
    def get_n_neurons(self):
        return self.module.weight.data.shape[0]

    def random_split(self, C_new):
        if C_new == 0:
            return 0, None
        C_out, C_in, kh, kw = self.module.weight.shape
        idx = np.random.choice(C_out, C_new)

        device = self.get_device()
        delta1 = F.normalize(torch.randn(C_new, C_in, kh, kw).to(device), p=2, dim=-1)
        delta2 = F.normalize(torch.randn(C_new, C_in, kh, kw).to(device), p=2, dim=-1)

        delta1 = delta1 * 1e-2
        delta2 = delta2 * 1e-2

        idx = torch.LongTensor(idx).to(device)

        new_layer = nn.Conv2d(in_channels=C_in,
                              out_channels=C_out+C_new,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)

        # for current layer
        new_layer.weight.data[:C_out, ...] = self.module.weight.data.clone()
        new_layer.weight.data[C_out:, ...] = self.module.weight.data[idx, ...]
        new_layer.weight.data[idx, ...] += delta1
        new_layer.weight.data[C_out:, ...] -= delta2

        if self.has_bias:
            new_layer.bias.data[:C_out, ...] = self.module.bias.data.clone()
            new_layer.bias.data[C_out:, ...] = self.module.bias.data[idx]

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm2d(C_out+C_new).to(device)
            new_bn.weight.data[:C_out] = self.bn.weight.data.clone()
            new_bn.weight.data[C_out:] = self.bn.weight.data[idx]
            new_bn.bias.data[:C_out] = self.bn.bias.data.clone()
            new_bn.bias.data[C_out:] = self.bn.bias.data[idx]
            new_bn.running_mean.data[:C_out] = self.bn.running_mean.data.clone()
            new_bn.running_mean.data[C_out:] = self.bn.running_mean.data[idx]
            new_bn.running_var.data[:C_out] = self.bn.running_var.data.clone()
            new_bn.running_var.data[C_out:] = self.bn.running_var.data[idx]
            self.bn = new_bn
        return C_new, idx

    def rdinit_grow_output(self):
        C_out, C_in, kh, kw = self.module.weight.shape
        device = self.get_device()
        new_layer = nn.Conv2d(in_channels=C_in,
                              out_channels=C_out+1,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)
        new_layer.weight.data[:C_out, ...] = self.module.weight.data.clone()
        self.module = new_layer

    def rdinit_grow_input(self):
        C_out, C_in, kh, kw = self.module.weight.shape
        device = self.get_device()
        new_layer = nn.Conv2d(in_channels=C_in+1,
                              out_channels=C_out,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)
        new_layer.weight.data[:,:C_in, ...] = self.module.weight.data.clone()
        self.module = new_layer

    def active_split(self, threshold):
        idx = torch.nonzero((self.w <= threshold).float()).view(-1)
        C_new = idx.shape[0]
        if C_new == 0:
            return 0, None

        C_out, C_in, kh, kw = self.module.weight.shape
        device = self.get_device()

        delta = self.v[idx, ...] * 1e-2

        delta = delta.view(C_new, C_in, kh, kw)

        new_layer = nn.Conv2d(in_channels=C_in,
                              out_channels=C_out+C_new,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)

        # for current layer
        new_layer.weight.data[:C_out, ...] = self.module.weight.data.clone()
        new_layer.weight.data[C_out:, ...] = self.module.weight.data[idx, ...]
        new_layer.weight.data[idx, ...] += delta
        new_layer.weight.data[C_out:, ...] -= delta

        if self.has_bias:
            new_layer.bias.data[:C_out, ...] = self.module.bias.data.clone()
            new_layer.bias.data[C_out:, ...] = self.module.bias.data[idx]

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm2d(C_out+C_new).to(device)
            new_bn.weight.data[:C_out] = self.bn.weight.data.clone()
            new_bn.weight.data[C_out:] = self.bn.weight.data[idx]
            new_bn.bias.data[:C_out] = self.bn.bias.data.clone()
            new_bn.bias.data[C_out:] = self.bn.bias.data[idx]
            new_bn.running_mean.data[:C_out] = self.bn.running_mean.data.clone()
            new_bn.running_mean.data[C_out:] = self.bn.running_mean.data[idx]
            new_bn.running_var.data[:C_out] = self.bn.running_var.data.clone()
            new_bn.running_var.data[C_out:] = self.bn.running_var.data[idx]
            self.bn = new_bn
        return C_new, idx

    def passive_split(self, idx):
        C_new = idx.shape[0]
        C_out, C_in, _, _ = self.module.weight.shape
        device = self.get_device()
        new_layer = nn.Conv2d(in_channels=C_in+C_new,
                              out_channels=C_out,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)

        new_layer.weight.data[:, :C_in, ...] = self.module.weight.data.clone()
        new_layer.weight.data[:, C_in:, ...] = self.module.weight.data[:, idx, ...] / 2.
        new_layer.weight.data[:, idx, ...] /= 2.
        if self.has_bias:
            new_layer.bias.data = self.module.bias.data.clone()
        self.module = new_layer

    def spffn_active_grow(self, threshold):
        idx = torch.nonzero((self.w <= threshold).float()).view(-1)

        C_out, C_in, kh, kw = self.module.weight.shape
        c1 = C_out - self.eout
        c3 = C_in - self.ein

        split_idx, new_idx = idx[idx < c1], idx[idx >= c1]
        n_split = split_idx.shape[0]
        n_new = new_idx.shape[0]

        c2 = c1 + n_split

        device = self.get_device()
        delta = self.v[split_idx, ...]

        new_layer = nn.Conv2d(in_channels=C_in,
                              out_channels=c1+n_split+n_new,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias).to(device)

        # for current layer [--original--c1--split_new--c2--add new--]
        old_W = self.module.weight.data.clone()

        try:
            old_W[:, C_in - self.ein:, :, :] = self.vni.clone()
        except:
            pass
        try:
            old_W[C_out-self.eout:, :C_in-self.ein, :, :] = self.vno.clone()
        except:
            pass

        new_layer.weight.data[:c1, ...] = old_W[:c1,...]

        if n_split > 0:
            new_layer.weight.data[c1:c2, ...] = old_W[split_idx, ...]
            new_layer.weight.data[split_idx,:c3,...] += delta
            new_layer.weight.data[c1:c2:,:c3,...] -= delta

        if n_new > 0:
            new_layer.weight.data[c2:, ...] = old_W[new_idx, ...]

        if self.has_bias:
            old_b = self.module.bias.data.clone()
            new_layer.bias.data[:c1, ...] = old_b[:c1,...].clone()
            if n_split > 0:
                new_layer.bias.data[c1:c2, ...] = old_b[split_idx]
            if n_new > 0:
                new_layer.bias.data[c2:,...] = 0.

        self.module = new_layer

        # for batchnorm layer
        if self.has_bn:
            new_bn = nn.BatchNorm2d(c1+n_split+n_new).to(device)
            new_bn.weight.data[:c1] = self.bn.weight.data[:c1].clone()
            new_bn.bias.data[:c1] = self.bn.bias.data[:c1].clone()
            new_bn.running_mean.data[:c1] = self.bn.running_mean.data[:c1].clone()
            new_bn.running_var.data[:c1] = self.bn.running_var.data[:c1].clone()

            if n_split > 0:
                new_bn.weight.data[c1:c2] = self.bn.weight.data[split_idx]
                new_bn.bias.data[c1:c2] = self.bn.bias.data[split_idx]
                new_bn.running_mean.data[c1:c2] = self.bn.running_mean.data[split_idx]
                new_bn.running_var.data[c1:c2] = self.bn.running_var.data[split_idx]

            if n_new > 0:
                new_bn.weight.data[c2:] = self.bn.weight.data[new_idx]
                new_bn.bias.data[c2:] = self.bn.bias.data[new_idx]
                new_bn.running_mean.data[c2:] = self.bn.running_mean.data[new_idx]
                new_bn.running_var.data[c2:] = self.bn.running_var.data[new_idx]
            self.bn = new_bn

        return n_split+n_new, split_idx, new_idx

    def spffn_passive_grow(self, split_idx, new_idx):
        n_split = split_idx.shape[0] if split_idx is not None else 0
        n_new = new_idx.shape[0] if new_idx is not None else 0

        C_out, C_in, _, _ = self.module.weight.shape
        if self.groups != 1:
            C_in = C_out
        device = self.get_device()
        c1 = C_in-self.ein
        if n_split == 0 and n_new == self.ein:
            return

        if self.groups != 1:
            self.groups = c1 + n_split + n_new
            C_out = self.groups
        new_layer = nn.Conv2d(in_channels=c1+n_split+n_new,
                              out_channels=C_out,
                              kernel_size=(self.kh, self.kw),
                              stride=(self.dh, self.dw),
                              padding=(self.ph, self.pw),
                              bias=self.has_bias, groups = self.groups).to(device)

        c2 = c1 + n_split

        if self.has_bias:
            new_layer.bias.data = self.module.bias.data.clone()

        if self.groups != 1:
            new_layer.weight.data[:c1,:,...] = self.module.weight.data[:c1,:,...].clone()
        else:
            new_layer.weight.data[:,:c1,...] = self.module.weight.data[:,:c1,...].clone()

        if n_split > 0:
            if self.groups == 1:
                new_layer.weight.data[:,c1:c2,:,:] = self.module.weight.data[:,split_idx,:,:] / 2.
                new_layer.weight.data[:,split_idx,...] /= 2.
            else:
                new_layer.weight.data[c1:c2, :,...] = self.module.weight.data[split_idx, :,...]
        if self.groups != 1:
            new_bn = nn.BatchNorm2d(C_out).to(device)
            out = C_out - n_new - n_split
            out1 = out + n_split
            out2 = out1 + n_new
            new_bn.weight.data[:out] = self.bn.weight.data.clone()[:out]
            new_bn.bias.data[:out] = self.bn.bias.data.clone()[:out]
            new_bn.running_mean.data[:out] = self.bn.running_mean.data.clone()[:out]
            new_bn.running_var.data[:out] = self.bn.running_var.data.clone()[:out]
            if n_split > 0:
                out1 = out + n_split
                new_bn.weight.data[out:out1] = self.bn.weight.data[split_idx]
                new_bn.bias.data[out:out1] = self.bn.bias.data[split_idx]
                new_bn.running_mean.data[out:out1] = self.bn.running_mean.data[split_idx]
                new_bn.running_var.data[out:out1] = self.bn.running_var.data[split_idx]

            if n_new > 0:
                new_bn.weight.data[out1:out2] = self.bn.weight.data[new_idx]
                new_bn.bias.data[out1:out2] = self.bn.bias.data[new_idx]
                new_bn.running_mean.data[out1:out2] = self.bn.running_mean.data[new_idx]
                new_bn.running_var.data[out1:out2] = self.bn.running_var.data[new_idx]
            self.bn = new_bn
        if n_new > 0:
            if self.groups != 1:
                new_layer.weight.data[c2:,:,...] = self.module.weight.data[new_idx, :,...]
            else:
                new_layer.weight.data[:,c2:,...] = self.module.weight.data[:,new_idx,...]

        self.module = new_layer
