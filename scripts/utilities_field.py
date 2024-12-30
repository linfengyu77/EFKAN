'''
modified utilities file
'''
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial


num_type = np.float32
complex_type = np.complex64
#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mt2d_Z(freq,E,H):
    #compute the impedance, apparent resistivity and phase of TE mode 2-D Magnetotellurics(MT) forward modeling problem
    omega = 2.0*np.pi*freq
    miu   =  4.0e-7*np.pi
    #compute the outputs
    Z = E/H
    rho = np.abs(Z)**2/(omega*miu)
    phs = np.arctan2(Z.imag, Z.real)*180.0/np.pi
        # phstm[i] = cm.phase(zyx[i])*180.0/np.pi

    return rho,phs
# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(num_type)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def read_complex(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(complex_type)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class GaussianNormalizer_out(object):
    ''' 
    multiple output
    '''
    def __init__(self, x, flag='Point', eps=0.00001):
        # self.n_out = x.size()[-1]
        self.eps = eps
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if flag == 'Point':
            self.mean = torch.mean(x,dim=[0,1,2],keepdim=True)
            self.std  = torch.std(x,dim=[0,1,2],keepdim=True)
            
        # elif flag == 'Unit':
        #     self.mean = torch.stack([torch.mean(x[...,i],0) for i in range(self.n_out)])
        #     self.std  = torch.stack([torch.std(x[...,i],0)  for i in range(self.n_out)])
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")

    def encode(self, x):
        mean = self.mean
        std  = self.std
        # notice: the input value would be changed
        x = (x - mean) / (std + self.eps)
        return x

    def decode(self, x,):
        mean = self.mean
        std  = self.std
        x = x*(std+self.eps)+mean
        return x

    def to(self,device):
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)

# normalization, pointwise gaussian
class GaussianNormalizer_minmax(object):
    ''' 
    multiple output
    '''
    def __init__(self, x, flag='Point', eps=0.00001):
        # self.n_out = x.size()[-1]
        self.eps = eps
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if flag == 'Point':
            self.min = torch.amin(x,dim=[0,1,2],keepdim=True)
            self.max = torch.amax(x,dim=[0,1,2],keepdim=True)
            
        # elif flag == 'Unit':
        #     self.mean = torch.stack([torch.mean(x[...,i],0) for i in range(self.n_out)])
        #     self.std  = torch.stack([torch.std(x[...,i],0)  for i in range(self.n_out)])
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")

    def encode(self, x):
        min = self.min
        max = self.max
        # notice: the input value would be changed
        x = 2*(x - min) / (max-min) - 1
        return x

    def decode(self, x,):
        min = self.min
        max = self.max
        x = (x+1)/2*(max-min)+min
        return x

    def to(self,device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
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


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

class LpLoss_out(object):
    ''' 
    multiple output
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):

        diff_norms = torch.linalg.norm(x - y, self.p,dim=[1,2])
        y_norms    = torch.linalg.norm(y, self.p, dim=[1,2])

        if self.reduction:
            if self.size_average:
                return torch.mean(torch.mean(diff_norms/y_norms,dim=-1))
            else:
                return torch.sum(torch.mean(diff_norms/y_norms,dim=-1))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

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

    def __call__(self, x, y):
        return self.rel(x, y)