'''
modified utilities file
'''
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
import os


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
    rho = torch.abs(Z)**2/(omega*miu)
    phs = torch.atan2(Z.imag, Z.real)*180.0/np.pi
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
# normalization, pointwise gaussian
class GaussianNormalizer_out(object):
    ''' 
    multiple output
    '''
    def __init__(self, x, flag='Point', eps=0.00001):
        self.n_out = x.size()[-1]
        self.eps = eps
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        if flag == 'Point':
            self.mean = torch.stack([torch.mean(x[...,i]) for i in range(self.n_out)])
            self.std  = torch.stack([torch.std(x[...,i])  for i in range(self.n_out)])
        elif flag == 'Unit':
            self.mean = torch.stack([torch.mean(x[...,i],0) for i in range(self.n_out)])
            self.std  = torch.stack([torch.std(x[...,i],0)  for i in range(self.n_out)])
        else:
            raise KeyError("invlid flag, must 'Point' or 'Unit'")

    def encode(self, x):
        for i in range(self.n_out):
            mean = self.mean[i,...]
            std  = self.std[i,...]
            # notice: the input value would be changed
            x[...,i] = (x[...,i] - mean) / (std + self.eps)
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
            x[...,i] = (x[...,i] * std[i,...]) + mean[i,...]
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
        # input doesn't reshape
        n_out = y.shape[-1]
        diff_norms = torch.norm(x - y,dim=[1,2])
        y_norms = torch.norm(y, dim=[1,2])

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)/n_out

        # return diff_norms/y_norms

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

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

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

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        # Sobolev norm可以用傅立叶变换乘以因子来计算（因为微分=傅立叶变换x因子）
        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

# copy from galerkin-transformer of Shuhao Cao
def showsolution(node, elem, u, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on triangulations
    '''
    markersize = 3000/len(node)

    if u.ndim == 1:  # (N, )
        uplot = ff.create_trisurf(x=node[:, 0], y=node[:, 1], z=u,
                                  simplices=elem,
                                  colormap="Viridis",  # similar to matlab's default colormap
                                  showbackground=True,
                                  show_colorbar=False,
                                  aspectratio=dict(x=1, y=1, z=1),
                                  )
        fig = go.Figure(data=uplot)

    elif u.ndim == 2 and u.shape[1] == 2:  # (N, 2)
        if u.shape[0] == elem.shape[0]:
            u /= (np.abs(u)).max()
            node = node[elem].mean(axis=1)

        uplot = ff.create_quiver(x=node[:, 0], y=node[:, 1],
                                 u=u[:, 0], v=u[:, 1],
                                 scale=.2,
                                 arrow_scale=.5,
                                 name='gradient of u',
                                 line_width=1,
                                 )

        fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()


def showsurf(x, y, z, **kwargs):
    '''
    show 2D solution either of a scalar function or a vector field
    on a meshgrid
    x, y, z: (M, N) matrix
    '''

    uplot = go.Surface(x=x, y=y, z=z,
                       colorscale='Viridis',
                       showscale=False),

    fig = go.Figure(data=uplot)

    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                          **kwargs)
    fig.show()


def showcontour(z, **kwargs):
    '''
    show 2D solution z of its contour
    '''
    uplot = go.Contour(z=z,
                       colorscale='RdYlBu',
                       line_smoothing=0.85,
                       line_width=0.1,
                       contours=dict(
                           coloring='heatmap',
                           #    showlabels=True,
                       )
                       )
    fig = go.Figure(data=uplot,
                    layout={'xaxis': {'title': 'x-label',
                                      'visible': False,
                                      'showticklabels': False},
                            'yaxis': {'title': 'y-label',
                                      'visible': False,
                                      'showticklabels': False}},)
    fig.update_traces(showscale=False)
    if 'template' not in kwargs.keys():
        fig.update_layout(template='plotly_dark',
                          margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    else:
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          **kwargs)
    fig.show()
    return fig

def plot_1d(fre, fre_list, ry, efno_pred, kan_pred, save_path, prefix):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fre_idx = np.where(fre_list == fre)
    # title = 'Frequency='+str(fre)+' Hz'
    label= ['FDM', 'EFNO', 'EFKAN']
    n_row,n_column = 2, 2
    fig, ax = plt.subplots(n_column, n_row)
    i = 0
    ylabels = [r"$log_{10}\rho_{xy}(\Omega \cdot m)$", r"$\Phi_{xy}(Degree)$", r"$log_{10}\rho_{yx}(\Omega \cdot m)$", r"$\Phi_{yx}(Degree)$"]
    for ii in range(n_row):
        for jj in range(n_column):
            ax[ii, jj].scatter(ry/1e3, efno_pred[i][0][fre_idx, :], color='k', marker='o', s=3, label=label[0]) 
            ax[ii, jj].scatter(ry/1e3, efno_pred[i][1][fre_idx, :], color='b', marker='^', s=3, label=label[1])
            ax[ii, jj].scatter(ry/1e3, kan_pred[i][1][fre_idx, :], color='r', marker='D', s=3, label=label[2])
            ax[ii, jj].set_ylabel(ylabels[i])
            ax[ii, jj].set_xlabel("Distance(km)")
            ax[ii, jj].legend()
            i += 1
    # fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path + "/1D_" + prefix + '_' + str(fre)+"Hz.pdf", bbox_inches='tight', pad_inches=0.05)