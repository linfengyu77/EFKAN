"""
to compare with deeponet

usage: python fourier_2d.py random_multi
"""

import os
import numpy as np
import torch
from torchinfo import summary
import yaml
from timeit import default_timer
import sys
sys.path.append("../scripts/") 
from utilities import *
from FNO import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

def get_batch_data(TRAIN_PATH, TEST_PATH, ntrain, ntest, r_train,s_train,r_test,s_test, batch_size,n_out):
    '''
    get and format data for training and testing

    Parameters:
    ----------
        - TRAIN_PATH : path to training data, e.g. '../Data/data/train_random.mat
        - TEST_PATH  : path to training data, e.g. '../Data/data/test_random.mat'
        - ntrain     : number of training data
        - ntest      : number of testing data
        - r_train    : downsampling factor of training data, [fac_input_x, fac_input_y,fac_output_x,fac_output_y]
        - s_train    : resolution of training data, [s_input_x, s_input_y, s_output_x, s_output_y]
        - r_test     : downsampling factor of testing data, [fac_input_x, fac_input_y,fac_output_x,fac_output_y]
        - s_test     : resolution of testing data, [s_input_x, s_input_y, s_output_x, s_output_y]
        - batch_size : batch size for training
        - n_out      : number of output channels, here is 4: rhoxy, phasexy, rhoyx, phaseyx
    '''
    print("begin to read data")
    key_map0 = ['rhoxy','phsxy','rhoyx','phsyx']
    key_map = key_map0[:n_out] # number of output channels
    t_read0 = default_timer()
    # get training data
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('sig')
    x_train = x_train[:ntrain,::r_train[0],::r_train[1]][:,:s_train[0],:s_train[1]]
    y_train = torch.stack([reader.read_field(key_map[i])\
    [:ntrain,::r_train[2],::r_train[3]][:,:s_train[2],:s_train[3]] for i in range(len(key_map))]).permute(1,2,3,0)
    freq_base    = reader.read_field('freq')[0]
    obs_base     = reader.read_field('obs')[0]
    freq    = torch.log10(freq_base[::r_train[2]][:s_train[2]]) # normalization
    obs     = obs_base[::r_train[3]][:s_train[3]]/torch.max(obs_base) # normalization
    loc1,loc2     = torch.meshgrid(freq,obs)
    # loc is the input of trunck net
    loc_train = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)
    del reader

    # get test data
    reader_test = MatReader(TEST_PATH)
    x_test = reader_test.read_field('sig')
    x_test = x_test[:ntest,::r_test[0],::r_test[1]][:,:s_test[0],:s_test[1]]
    y_test = torch.stack([reader_test.read_field(key_map[i])\
    [:ntest,::r_test[2],::r_test[3]][:,:s_test[2],:s_test[3]] for i in range(len(key_map))]).permute(1,2,3,0)
    freq    = torch.log10(freq_base[::r_test[2]][:s_test[2]])
    obs     = obs_base[::r_test[3]][:s_test[3]]/torch.max(obs_base)
    loc1,loc2= torch.meshgrid(freq,obs)
    loc_test = torch.cat((loc1.reshape(-1,1),loc2.reshape(-1,1)),-1)
    del reader_test

    #data normalization
    x_normalizer = GaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = GaussianNormalizer_out(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain,s_train[0],s_train[1],1)
    x_test = x_test.reshape(ntest,s_test[0],s_test[1],1)

    # Convert data to dataloader  
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    t_read1 = default_timer()
    print(f"reading finished in {t_read1-t_read0:.3f} s")

    return loc_train,loc_test,train_loader, test_loader, x_normalizer,y_normalizer

def print_model(model, flag=True):
    if flag:
        summary(model)
    # else:
    #     print("do not print model")

def batch_train(model, train_loader, y_normalizer, loss_func, optimizer, scheduler, device):
    '''min-batch training
    Parameters:
    -----------
        - model       : neural network model
        - batch_size  : batch size
        - s_train     : resolution of training data
        - loc         : location of training data
        - train_loader: dataloader for training data
        - y_normalizer: normalizer for training output data
        - loss_func   : function for loss
        - optimizer   : optimizer
        - scheduler   : scheduler
        - device      : device of data and model storage, 'cpu' or 'cuda:id'
    '''
    # if not y_normalizer.is_cuda:
    #     y_normalizer.to(device)
    train_l2 = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) # input (batch, s, s,1)

        # batch_size = len(x)
        optimizer.zero_grad()
        out = model(x)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()        
        train_l2 += loss.item()
    scheduler.step()
    return train_l2


def batch_validate(model,test_loader, y_normalizer, loss_func, device):
    '''batch validation of test data
    Parameters: as batch_train function
    '''
    # if not y_normalizer.is_cuda:
    #     y_normalizer.to(device)
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # batch_size = len(x)
            # n_out = y.shape[-1]
            out = model(x)#.reshape(batch_size, s_test[2], s_test[3],-1)
            out = y_normalizer.decode(out)
            test_l2 += loss_func(out, y).item()
    return test_l2

def run_train(model,train_loader, test_loader,
              y_normalizer,loss_func, optimizer, scheduler,epochs, 
              thre_epoch, patience,save_step, 
              save_mode, model_path,model_path_temp,
              ntrain, ntest,  device,log_file):
    '''
    training and validation of model

    Parameters: some parameters are same as 'batch_train' function and 'get_batch_data' function
    -----------
        - epochs: number of epochs
        - thre_epoch: threshold of epochs for early stopping
        - patience: patience epochs that loss continue to rise. ig loss continue to rise for 'patience' epochs, stop early
        - save_step: save model every 'save_step' epochs
        - save_mode: save whole model or static dictionary
        - model_path: path to save model
        - model_path_temp: path to save temporary model
        - log_file: path to save log file
    '''
    
    val_l2 = np.inf
    stop_counter = 0

    temp_file = None
    for ep in range(epochs):
        t1 = default_timer()
        model.train()
        train_l2 = batch_train(model, train_loader, y_normalizer,  loss_func, optimizer, scheduler, device)
        model.eval()
        test_l2 = batch_validate(model, test_loader, y_normalizer, loss_func, device)
        train_l2/= ntrain
        test_l2 /= ntest

        # save model. Make sure that the model has been saved even if you stop the program manually. 
        # This is useful when you find that the training epoch is enough to stop the program
        if ep % save_step == 0:
            # Delete the previous saved model 
            if temp_file is not None:
                os.remove(temp_file)
            # only save static dictionary instead of whole model
            torch.save(model.state_dict(), model_path_temp+'_epoch_'+str(ep)+'.pkl')
            temp_file = model_path_temp+'_epoch_'+str(ep)+'.pkl'

        # early stop
        if ep > thre_epoch:
            if test_l2 < val_l2:
                # val_epoch = ep
                val_l2 = test_l2
                stop_counter = 0 
                if save_mode == 'state_dict':
                    torch.save(model.state_dict(), model_path+'.pkl')
                else: # save whole model, not recommended.
                    torch.save(model, model_path+'.pt')
            else:
                stop_counter += 1
            # If the error continues to rise within 'patience' epochs, break
            if stop_counter > patience: 
                print(f"Early stop at epoch {ep}")
                print(f"# Early stop at epoch {ep}",file=log_file)
                break

        t2 = default_timer()
        print(ep, t2-t1, train_l2, test_l2)
        print(ep, t2-t1, train_l2, test_l2,file=log_file)


################################################################
# configs
################################################################
def main(item):
    '''
    item: item name in yaml file
    '''
    t0 = default_timer()
    # read configurations from yaml file
    with open( 'config.yml') as f:
        config = yaml.full_load(f)
    config = config[item]
    cuda_id = "cuda:"+str(config['cuda_id'])
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
    TRAIN_PATH = config['TRAIN_PATH']
    TEST_PATH  = config['TEST_PATH']
    save_mode  = config['save_mode']
    save_step  = config['save_step']
    n_out      = config['n_out'] # rhoxy,phsxy,rhoyx,phsyx
    model_path = "../model/"+config['name']+ "_"+str(n_out) # save path and name of model
    model_path_temp = "../temp/"+config['name']+"_"+ str(n_out)
    log_path = "../Log/"+config['name']+"_"+str(n_out)+'.log'
    ntrain = config['ntrain']
    ntest  = config['ntest']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    step_size = config['step_size']
    gamma = config['gamma']
    modes = config['modes']
    width = config['width']
    s_train = config['s_train']
    r_train = config['r_train']
    s_test = config['s_test']
    r_test = config['r_test']
    layer_num = config['layer_num']
    last_size = config['last_size']
    layer_sizes = config['layer_sizes'] + [s_train[0]*s_train[1]]
    act_fno   = config['act_fno']
    act_func  = config['act_func']
    init_func = config['init_func']    
    patience = config['patience'] # if there is {patience} epoch that val_error is larger, early stop,
    thre_epoch = config['thre_epoch']# condiser early stop after {thre_epoch} epochs
    print_model_flag = config['print_model_flag'] # print model

    ################################################################
    # load data and data normalization
    ################################################################
    _,_,train_loader, test_loader, _,y_normalizer = \
        get_batch_data(TRAIN_PATH, TEST_PATH, ntrain, ntest,\
                        r_train, s_train,r_test,s_test,batch_size,n_out)
    y_normalizer.to(device)

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width,\
        n_out,layer_num, last_size, act_fno).to(device)
    # print(count_params(model))
    print_model(model, print_model_flag)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)

    log_file = open(log_path,'a+')
    print("####################")
    print("begin to train model")
    run_train(model, train_loader, test_loader, y_normalizer, myloss, optimizer, scheduler, epochs, \
               thre_epoch, patience, save_step,save_mode, model_path,model_path_temp, ntrain, ntest, device,log_file)

    tn = default_timer()
    print(f'all time:{tn-t0:.3f}s')
    print(f'# all time:{tn-t0:.3f}s',file=log_file)
    log_file.close()


if __name__ == '__main__':
    # item name in yaml config file
    try:
        item = sys.argv[1]
    except: # usefule in vscode debug mode
        item = 'random_fno'
    main(item)