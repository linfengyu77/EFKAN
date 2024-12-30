"""
FCN, modified from (Zhu,2019) : https://github.com/cics-nd/pde-surrogate.git

"""
import numpy as np
import torch
import torch.optim as optim

import sys
sys.path.append('../')
import sys
sys.path.append("../scripts/") 
from E_codec import EDenseED
from load import load_E_data
from data_process import LpLoss
from timeit import default_timer
import os


import random
import yaml

def _batch_train(train_loader, y_normalizer,model,optimizer,scheduler,loss_func,epoch,total_steps,device):
    train_l2 = 0.0
    for batch_idx, (input,loc, y) in enumerate(train_loader, start=1):
        input, loc, y = input.to(device), loc[0].to(device), y.to(device)
        batch_size = len(input)
        nz, ny= y.shape[-1],y.shape[-2]
        input_size = nz* ny
        model.zero_grad()
        out = model(input,loc)
        n_out = y.shape[1]
        # for muliti-output, restore one dimesion to (H,W) at each channel and then concatenate
        out = torch.cat(([out[:,i*input_size:(i+1)*input_size].reshape(batch_size,-1,nz,ny) for i in range(n_out)]),1)

        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)
        loss = loss_func(out, y)
        loss.backward()
        # lr scheduling
        # step = (epoch - 1) * len(train_loader) + batch_idx
        # pct = step / total_steps
        # lr = scheduler.step(pct)
        # adjust_learning_rate(optimizer, lr)
        optimizer.step()
        train_l2 += loss.item()
    scheduler.step()

    return train_l2

def _batch_test(test_loader,y_normalizer,model,loss_func,device):
    # model.eval()
    test_l2 = 0.0
    for batch_idx, (input, loc,y) in enumerate(test_loader):
        input, loc, y = input.to(device), loc[0].to(device),  y.to(device)
        batch_size = len(input)
        nz,ny = y.shape[-1],y.shape[-2]
        input_size = nz * ny
        out = model(input,loc)
        n_out = y.shape[1]
        # for muliti-output, restore one dimesion to (H,W) at each channel and then concatenate
        out = torch.cat(([out[:,i*input_size:(i+1)*input_size].reshape(batch_size,-1,nz,ny) for i in range(n_out)]),1)

        out = y_normalizer.decode(out)
        loss = loss_func(out,y)
        test_l2 += loss.item()
    return test_l2

def train(train_loader, test_loader,y_normalizer,model,optimizer,scheduler,loss_func,config,device):


    model_path =  config['model_path']
    model_path_temp =  config['model_path_temp']
    save_mode = config['save_mode']
    patience = config['patience']
    save_step  = config['save_step']
    thre_epoch = config['thre_epochs']
    log_path   =  config['log_path']
    log_file = open(log_path,'a+')

    val_l2 = np.inf
    stop_counter = 0
    temp_file = None

    print('Start training...................................................')
    start_epoch = 1 
    tic = default_timer()
    # step = 0
    total_steps = config['epochs'] * len(train_loader)
    print(f'total steps: {total_steps}')
    for epoch in range(start_epoch, config['epochs'] + 1):
        ep =epoch
        t1 = default_timer()
        model.train()
        train_l2 = _batch_train(train_loader, y_normalizer,model,optimizer,scheduler,loss_func,epoch,total_steps,device)
        train_l2 = train_l2 / config['ntrain']
        
        with torch.no_grad():
            model.eval()
            test_l2 = _batch_test(test_loader,y_normalizer,model,loss_func,device)
            test_l2 = test_l2 / config['ntest']

        if epoch % save_step == 0:
            # Delete the previous saved model 
            if temp_file is not None:
                os.remove(temp_file)
            # only save static dictionary instead of whole model
            torch.save(model.state_dict(), model_path_temp+'_epoch_'+str(ep)+'.pkl')
            temp_file = model_path_temp+'_epoch_'+str(ep)+'.pkl'

        # early stop
        if epoch > thre_epoch:
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
        print(epoch, t2-t1, train_l2, test_l2)
        print(epoch, t2-t1, train_l2, test_l2,file=log_file)

    tic2 = default_timer()

    print(f'all time:{tic2 - tic:.3f}s')
    print(f'# all time:{tic2 - tic:.3f}s',file=log_file)
    log_file.close()

def main(item):

    # args = Parser().parse()
    with open( 'config_CNN.yml') as f:
        config = yaml.full_load(f)
    config = config[item]
    config['layer_sizes'] =  config['layer_sizes'] + [64*64]
    config['model_path'] = "../model/"+item
    config['model_path_temp'] = "../temp/"+item
    config['log_path'] = "../Log/"+item+'.log'

    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    device = torch.device(f"cuda:{config['cuda']}" if torch.cuda.is_available() else "cpu")

    train_file =  config['train_file']  
    test_file  =  config['test_file']

    loss_func = LpLoss(size_average=False)

    model = EDenseED(config['layer_sizes'], 
                    config['act_func'], config['init_func'],
                    in_channels=1, out_channels=4, 
                    imsize=config['imsize'],
                    blocks=config['blocks'],
                    growth_rate  =config['growth_rate'],
                    init_features=config['init_features'],
                    drop_rate    =config['drop_rate'],
                    out_activation=None,
                    upsample=config['upsample'])

    model = model.to(device)
    # load data
    train_loader, test_loader, _,y_normalizer = load_E_data(train_file,test_file,config['ntest'],config['ntest'],config['batch_size'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                        weight_decay=config['weight_decay'])
    # scheduler = OneCycleScheduler(lr_max=config['lr'], div_factor=config['lr_div'], pct_start=config['lr_pct'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    train(train_loader, test_loader,y_normalizer,model,optimizer,scheduler,loss_func,config,device)   

if __name__ == '__main__':
    # item name in yaml config file
    try:
        item = sys.argv[1]
    except: # usefule in vscode debug mode
        item = 'E_best'
    main(item)
