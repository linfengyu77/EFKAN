import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.io as scio
import torch
import torch.nn as nn
from torchinfo import summary
from timeit import default_timer
import sys
sys.path.append("../run/")
import yaml
from utilities import *
from efno_mlp import *

plt.rcParams['font.family']       = 'arial' # 'Times New Roman' #
# plt.rcParams['axes.linewidth']    = 1
plt.rcParams['xtick.major.size']  = 2.
plt.rcParams['ytick.major.size']  = 2.5
plt.rcParams['xtick.minor.size']  = 1.5
plt.rcParams['ytick.minor.size']  = 1.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['lines.linewidth']   = 1.5
plt.rcParams['lines.markersize']  = 3.5
plt.rcParams['font.size']    = 10
# plt.rcParams['figure.titlesize'] = 2
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.labelsize']  = 7 # x, y label size
plt.rcParams['axes.titlesize'] = 7 # font size of the axes title
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['legend.title_fontsize'] = 7
plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.pad_inches"] = 0.1
plt.rcParams['image.cmap'] = 'jet_r'
plt.rcParams['figure.dpi'] = 150
cm = 1/2.54

class EFNO_Testing(object):
    def __init__(self, config_yaml, item):
        with open(config_yaml) as f:
            config = yaml.full_load(f)
        config = config[item]
        cuda_id = "cuda:"+str(config['cuda_id'])
        self.device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
        self.TRAIN_PATH = config['TRAIN_PATH']
        self.TEST_PATH  = config['TEST_PATH']
        self.save_mode  = config['save_mode']
        self.save_step  = config['save_step']
        self.n_out      = config['n_out'] # rhoxy,phsxy,rhoyx,phsyx
        self.model_path = "../model/"+config['name']+ "_"+str(self.n_out) # save path and name of model
        self.model_path_temp = "../temp/"+config['name']+"_"+ str(self.n_out)
        self.log_path = "../Log/"+config['name']+"_"+str(self.n_out)+'.log'
        self.ntrain = config['ntrain']
        self.ntest  = config['ntest']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.step_size = config['step_size']
        self.gamma = config['gamma']
        self.modes = config['modes']
        self.width = config['width']
        self.s_train = config['s_train']
        self.r_train = config['r_train']
        self.s_test = config['s_test']
        self.r_test = config['r_test']
        self.layer_num = config['layer_num']
        self.last_size = config['last_size']
        self.layer_sizes = config['layer_sizes'] + [self.s_train[0] * self.s_train[1]]
        self.act_fno   = config['act_fno']
        self.act_func  = config['act_func']
        self.init_func = config['init_func']    
        self.patience = config['patience'] # if there is {patience} epoch that val_error is larger, early stop,
        self.thre_epoch = config['thre_epoch']# condiser early stop after {thre_epoch} epochs
        self.print_model_flag = config['print_model_flag'] 

        self.data   = scio.loadmat(self.TEST_PATH)
        self.zn     = self.data['zn'][0][::self.r_test[0]][:self.s_test[0]+1]
        self.yn     = self.data['yn'][0][::self.r_test[1]][:self.s_test[1]+1]
        self.freq   = self.data['freq'][0][::self.r_test[2]][:self.s_test[2]]
        self.ry     = self.data['obs'][0][::self.r_test[3]][:self.s_test[3]]
        self.loc_train, self.loc_test, self.train_loader,\
        self.test_loader, self.x_normalizer, self.y_normalizer = get_batch_data(self.TRAIN_PATH, self.TEST_PATH, \
                                                                    self.ntrain, self.ntest, self.r_train, \
                                                                    self.s_train, self.r_test, self.s_test, \
                                                                    self.batch_size, self.n_out)
        if os.path.exists(self.model_path+'.pt'):
            self.model=torch.load(self.model_path+'.pt',map_location=self.device)
        elif os.path.exists(self.model_path+'.pkl'):
            self.model = deeponet(self.layer_sizes, self.act_func, self.init_func, self.modes, self.modes, self.width,\
                self.n_out, self.layer_num, self.last_size, self.act_fno).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path+'.pkl',map_location=self.device))
        else:
            raise RuntimeError('no model file')
        
        self.lossfn = LpLoss(size_average=False)
 
    def error_r(self, x1, x0):
        return np.linalg.norm(x1-x0)/np.linalg.norm(x0)
    
    def evaluate(self, rho_id, save_path, prefix):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.model.eval()
        self.model.to(self.device)
        self.y_normalizer.to(self.device)
        test_l2 = 0.0
        t1 = default_timer()
        loc_test = self.loc_test.to(self.device)
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(loc_test, x)
                input_size = self.s_test[2] * self.s_test[3]
                n_out = y.shape[-1]
                out = torch.cat(([out[:,i*input_size:(i+1)*input_size].reshape(self.batch_size, self.s_test[2], self.s_test[3],-1) \
                        for i in range(n_out)]),-1)
                out = self.y_normalizer.decode(out)
                test_l2 += self.lossfn(out, y).item()
        test_l2 /= self.ntest
        t2 = default_timer()
        self.total_time = t2-t1
        self.average_loss = test_l2
    
        n_bins = 5 # number of ticks in colorbar
        a_cof = self.x_normalizer.decode(x)[rho_id, ..., 0].cpu().numpy()
        a_cof = np.log10(1/(10**a_cof))
        rhoxy = out[rho_id, ...,0].cpu().numpy()
        phsxy = -1*out[rho_id, ...,1].cpu().numpy()
        rhoyx = out[rho_id, ...,2].cpu().numpy()
        phsyx = -1*out[rho_id, ...,3].cpu().numpy()+180
        rhoxy_true = y[rho_id, ...,0].cpu().numpy()
        phsxy_true = -1*y[rho_id, ...,1].cpu().numpy()
        rhoyx_true = y[rho_id, ...,2].cpu().numpy()
        phsyx_true = -1*y[rho_id, ...,3].cpu().numpy()+180

        self.error_rhoxy = self.error_r(rhoxy,rhoxy_true)
        self.error_rhoyx = self.error_r(rhoyx,rhoyx_true)
        self.error_phsxy = self.error_r(phsxy,phsxy_true)
        self.error_phsyx = self.error_r(phsyx,phsyx_true)
        
        fig = plt.figure(figsize=(10*cm,4*cm))
        ax = plt.subplot(1,1,1)
        h=ax.pcolormesh(self.yn/1e3, self.zn/1e3,a_cof, vmin=0,vmax=4,shading='flat')#,edgecolors='k')
        ax.set_xlabel('y (km)')
        ax.set_ylabel('z (km)')
        ax.invert_yaxis()
        cbar = fig.colorbar(h)
        cbar.set_label(r"$\rho$ / $\log_{10}(\Omega\cdot m)$")
        cbar.ax.locator_params(nbins=n_bins)
        ax.set_aspect(1)
        plt.savefig(save_path+'/res_'+str(rho_id)+'_'+str(prefix)+".pdf", bbox_inches='tight',pad_inches=0.05)

        obs_id = 35
        f_id = 40    # idx of frequency
        id_f = 40
        n_row,n_column = 4, 3
        figsize   = (5*n_column*cm,3.5*n_row*cm)
        labelpad       = 6 # label pad between label and bar (value is relative to axis value?)  
        text_size      = 7 # text size
        bar_size       = 7 # label size of colorbar
        # for line style
        color          = 'white'
        linestyle      = 'dashed'
        color_f        = 'black'
        linestyle_f    = 'solid'
        ticks_position = 'bottom' # colorbar ticks location relative to bar
        loc            = 'right' # colorbar location relative to map
        text_x_pos     = -135 # text location (value is relative to axis data in map)
        text_y_pos     = 18.0 
        rotation       = 0  # text rotation
        shading        = 'nearest' # pcolor shading, nearest for same size of x, y, C.

        plt.rcParams['axes.titlesize'] = 6 # font size of the axes title
        title_out = ['FDM','ENFO','Error']
        bar_label = [r'$\log_{10}\,\rho_{xy}\,(\Omega m)$',r'$\phi_{xy}\,$(degree)',\
                    r'$\log_{10}\,\rho_{yx}\,(\Omega m)$',r'$\phi_{yx}\,$(degree)']
        x_label   = r'distance$\,$(km)'
        y_label   = r'frequency$\,$(Hz)'
        rhoxy_error = rhoxy_true-rhoxy
        phsxy_error = phsxy_true-phsxy
        rhoyx_error = rhoyx_true-rhoyx
        phsyx_error = phsyx_true-phsyx
        data = [[rhoxy_true,rhoxy],
                [phsxy_true,phsxy],
                [rhoyx_true,rhoyx],
                [phsyx_true,phsyx]]
        
        data_error=[[rhoxy_error/rhoxy_true],
                    [phsxy_error/phsxy_true],
                    [rhoyx_error/rhoyx_true],
                    [phsyx_error/phsyx_true]]

        norm_rhoxy = colors.Normalize(vmin=np.min(data[0]),vmax=np.max(data[0]))
        norm_phsxy = colors.Normalize(vmin=np.min(data[1]),vmax=np.max(data[1]))
        norm_rhoyx = colors.Normalize(vmin=np.min(data[2]),vmax=np.max(data[2]))
        norm_phsyx = colors.Normalize(vmin=np.min(data[3]),vmax=np.max(data[3]))
        norm       = [norm_rhoxy,norm_phsxy,norm_rhoyx,norm_phsyx]

        ry_row      = [self.ry[obs_id]/1e3, self.ry[obs_id]/1e3]
        ry_column   = [np.min(self.ry)/1e3, np.max(self.ry)/1e3]
        freq_row    = [np.min(self.freq), np.max(self.freq)]
        freq_column = [self.freq[f_id], self.freq[f_id]]
        ry_f        = ry_column
        freq_f      = [self.freq[id_f], self.freq[id_f]]

        labelpad        = [3,4,7,4] # label pad between label and bar (value is relative to axis value?)  
        labelpad_error  = [3,3,3,3] #label pad between label and bar (value is relative to axis value?) 
        pad            = 0.02 # pad between colorbar and map (value is relative to figsize?)   

        fig,ax     = plt.subplots(n_row,n_column,figsize=figsize)
        for j in range(n_column):
            for i in range(n_row):
                if j!=n_column-1:
                    h = ax[i,j].pcolormesh(self.ry/1e3, self.freq,data[i][j],norm=norm[i], shading=shading)
                else: 
                    h = ax[i,j].pcolormesh(self.ry/1e3, self.freq,data_error[i][0], shading=shading)
                    
                ax[i,j].set_yscale("log")
                    
                if i == 0:
                    ax[i,j].set_title(title_out[j])
                if i!=n_row-1:
                    ax[i,j].set_xticks([])
                    
                if j == 0 :
                    ax[i,j].yaxis.tick_left()
                    ax[i,j].yaxis.set_label_position("left")
                    ax[i,j].set_ylabel(y_label)
                elif j==1:
                    ax[i,j].set_yticks([])               
                if j == n_column-2:
                    cbar = fig.colorbar(h,ax=[ax[i,jj] for jj in range(n_column-1)],location=loc,pad=pad)
                    cbar.set_label(bar_label[i],fontsize=bar_size,labelpad=labelpad[i])
                    cbar.ax.xaxis.set_ticks_position(ticks_position)
                elif j==n_column-1:
                    cbar = fig.colorbar(h,ax=ax[i,j] ,location=loc,pad=pad+0.025)
                    cbar.set_label('relative error',fontsize=bar_size,labelpad=labelpad_error[i])
                    cbar.ax.xaxis.set_ticks_position(ticks_position)                                
            ax[i,j].set_xlabel(x_label)            
        plt.savefig(save_path+'/efno_{}_rho_{}_total_time_{:.4f}_average_loss_{:.4f}_rho_xy_error_{:.4f}\
                    _rho_yx_error_{:.4f}_phs_xy_error_{:.4f}_phs_yx_error_{:.4f}.pdf'.format(prefix, str(rho_id),\
                    self.total_time, self.average_loss, self.error_rhoxy, self.error_rhoyx,\
                    self.error_phsxy, self.error_phsyx),bbox_inches='tight',pad_inches=0.05)

        return data, data_error