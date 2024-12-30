"""
Load args and model from a directory
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
import scipy.io as scio
import json
from timeit import default_timer
from data_process import GaussianNormalizer, GaussianNormalizer_out


def load_args(run_dir):
    with open(run_dir + '/args.txt') as args_file:  
        args = Namespace(**json.load(args_file))
    # pprint(args)
    return args


def load_data(train_file, test_file,n_train,n_test, batch_size):

    print("begin to read data")
    key_map = ['rhoxy','phsxy','rhoyx','phsyx']
    t_read0 = default_timer()
    # get training data
    data = scio.loadmat(train_file)
    x_train = data['sig']
    y_train = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=1)
    x_train = torch.FloatTensor(x_train[:n_train])
    y_train = torch.FloatTensor(y_train[:n_train])
    # get testing data
    data = scio.loadmat(test_file)
    x_test = data['sig']
    y_test = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=1)
    x_test = torch.FloatTensor(x_test[:n_test])
    y_test = torch.FloatTensor(y_test[:n_test])

    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = GaussianNormalizer_out(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train),batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TensorDataset(x_test, y_test),batch_size=batch_size, shuffle=False)
    t_read1 = default_timer()
    print(f"reading finished in {t_read1-t_read0:.3f} s")
    return train_loader, test_loader, x_normalizer,y_normalizer

def load_E_data(train_file, test_file,n_train,n_test, batch_size):

    print("begin to read data")
    key_map = ['rhoxy','phsxy','rhoyx','phsyx']
    t_read0 = default_timer()
    # get training data
    data = scio.loadmat(train_file)
    x_train = data['sig']
    y_train = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=1)
    x_train = torch.FloatTensor(x_train[:n_train])
    y_train = torch.FloatTensor(y_train[:n_train])

    freq_base    = data['freq'][0]
    obs_base     = data['obs'][0]
    freq_base    = torch.FloatTensor(freq_base)
    obs_base     = torch.FloatTensor(obs_base)
    freq    = torch.log10(freq_base) # normalization
    obs     = obs_base/torch.max(obs_base) # normalization
    loc1,loc2     = torch.meshgrid(freq,obs)
    # loc is the input of trunck net
    loc_train = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)

    # get testing data
    data = scio.loadmat(test_file)
    x_test = data['sig']
    y_test = np.stack([data[key_map[i]] for i in range(len(key_map))], axis=1)
    x_test = torch.FloatTensor(x_test[:n_test])
    y_test = torch.FloatTensor(y_test[:n_test])

    # freq    = torch.log10(freq_base) # normalization
    # obs     = obs_base/torch.max(obs_base) # normalization
    # loc1,loc2     = torch.meshgrid(freq,obs)
    # loc is the input of trunck net
    loc_test = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)

    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_normalizer = GaussianNormalizer_out(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(x_train, loc_train.repeat(n_train,1,1),y_train),batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(TensorDataset(x_test,  loc_test.repeat(n_test,1,1), y_test),batch_size=batch_size, shuffle=False)
    t_read1 = default_timer()
    print(f"reading finished in {t_read1-t_read0:.3f} s")
    return train_loader, test_loader, x_normalizer,y_normalizer