'''
拓展的CNN，类似EFNO
'''

from codec import DenseED
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

# activattion type
act_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "idt": nn.Identity(),
    "gelu": nn.GELU()
}

# initiation method
init_dict={
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "uniform": nn.init.uniform_,
    "norm": nn.init.normal_
}


class FNN(nn.Module):
    def __init__(self, layer_sizes, act_func, init_func):
        super(FNN, self).__init__()
        """
        Fully-connected neural network.

        Parameters:
        -----------
            - layer_sizes : list of integers, each integer is the size of a layer
                          : last element must be H*W of input data.
            - act_func    : activation function, key must in act_dict
            - init_func   : initialization function, key must in init_dict

        input(xy,2)
        output(xy, n_out*(H*W))
        """

        if act_func in act_dict.keys():
            self.activation = act_dict[act_func]
        else:
            raise KeyError("act name not in act_dict")
        if init_func in init_dict.keys():
            initializer = init_dict[init_func]
        else:
            raise KeyError("init name not in init_dict")

        self.linears = nn.ModuleList()
        self.BN = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            # self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.linears.append(KAN([layer_sizes[i - 1], layer_sizes[i]]))
            initializer(self.linears[-1].weight)
            nn.init.zeros_(self.linears[-1].bias)
            self.BN.append(nn.BatchNorm1d(layer_sizes[i]))

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = self.activation(self.linears[i](x))
            x = self.BN[i](x)
        x = self.linears[-1](x)
        return x


class EDenseED(nn.Sequential):
    def __init__(self, layer_sizes, act_func, init_func,
                 in_channels, out_channels, imsize, blocks, growth_rate=16,
                 init_features=48, drop_rate=0, bn_size=8, bottleneck=False, 
                 out_activation=None, upsample='nearest'):
        '''
        parameters like in DenseED and FNN 
        '''
        super(EDenseED, self).__init__()

        self.fnn = FNN(layer_sizes, act_func, init_func)
        self.cnn = DenseED(in_channels, out_channels, imsize, blocks, \
                            growth_rate,init_features, drop_rate, bn_size, bottleneck, 
                            out_activation, upsample)

    def forward(self, x, loc):
        x1 = self.cnn(x)
        n_batch = x1.shape[0]
        n_out   = x1.shape[1]
        # for multi-output, reshape output of each channel to 1-dimension respectively, and then concatenate
        # it's wrong to reshape directly (because you need to restore to the original shape)
        x1 = [x1[:,i,...].view(n_batch,-1) for i in range(n_out)]
        x2 = self.fnn(loc)
        # for multi-output, multiply output of branch net to the that of trunck net for each channel
        x = [torch.einsum("bi,ni->bn",x1[i],x2) for i in range(n_out)]   
        x = torch.cat(x,dim=-1)
        return x

class EDenseED0(nn.Sequential):
    def __init__(self, layer_sizes, act_func, init_func,
                 in_channels, out_channels, imsize, blocks, growth_rate=16,
                 init_features=48, drop_rate=0, bn_size=8, bottleneck=False, 
                 out_activation=None, upsample='nearest'):
        '''
        parameters like in DenseED and FNN 
        '''
        super(EDenseED0, self).__init__()

        self.fnn = FNN(layer_sizes, act_func, init_func)
        self.cnn = DenseED(in_channels, out_channels, imsize, blocks, \
                            growth_rate,init_features, drop_rate, bn_size, bottleneck, 
                            out_activation, upsample)
        # self.fc1 = nn.Linear(4, 128)
        self.fc1 = KAN([4, 128])
        self.activation =nn.GELU()
        # notice!
        # self.fc2 = nn.Linear(128, 4)
        self.fc2 = KAN([128, 4])

    def forward(self, x, loc):
        x = self.cnn(x).permute(0,2,3,1)
        x = self.activation(self.fc1(x))
        x1 = self.fc2(x).permute(0,3,1,2)
        n_batch = x1.shape[0]
        n_out   = x1.shape[1]
        # for multi-output, reshape output of each channel to 1-dimension respectively, and then concatenate
        # it's wrong to reshape directly (because you need to restore to the original shape)
        x1 = [x1[:,i,...].view(n_batch,-1) for i in range(n_out)]
        x2 = self.fnn(loc)
        # for multi-output, multiply output of branch net to the that of trunck net for each channel
        x = [torch.einsum("bi,ni->bn",x1[i],x2) for i in range(n_out)]   
        x = torch.cat(x,dim=-1)
        return x

