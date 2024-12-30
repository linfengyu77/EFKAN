import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce

class GaussianNormalizer_out(object):
    ''' 
    multiple output
    '''
    def __init__(self, x, flag='Point', eps=0.00001):
        self.n_out = x.size()[1]
        self.eps = eps
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if flag == 'Point':
            self.mean = torch.stack([torch.mean(x[:,i,...]) for i in range(self.n_out)])
            self.std  = torch.stack([torch.std( x[:,i,...])  for i in range(self.n_out)])
        elif flag == 'Unit':
            self.mean = torch.stack([torch.mean(x[:,i,...],0) for i in range(self.n_out)])
            self.std  = torch.stack([torch.std( x[:,i,...],0)  for i in range(self.n_out)])
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")

    def encode(self, x):
        for i in range(self.n_out):
            mean = self.mean[i,...]
            std  = self.std[i,...]
            # notice: the input value would be changed
            x[:,i,...] = (x[:,i,...] - mean) / (std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        for i in range(self.n_out):
            x[:,i,...] = (x[:,i,...] * std[i,...]) + mean[i,...]
        return x

    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
# normalization, pointwise gaussian

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, flag='Point', eps=0.00001):
        self.eps = eps
        # 每个样本的mean与位置无关
        if flag == 'Point':
            self.mean = torch.mean(x)
            self.std = torch.std(x)
        elif flag == 'Unit':
            self.mean = torch.mean(x, 0)
            self.std = torch.std(x, 0)
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self,x,y):
        return self.rel(x,y)