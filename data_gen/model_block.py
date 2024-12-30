'''
generate random model for coarse grid
usage: python model_gen.py 2000 40 train_gen

how to expand boundaries? use the same sigma near boundary.

random + blocks
'''
import ray
import gaussian_random_fields as gr
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import random
import sys
import numpy as np
import scipy.io as scio
from timeit import default_timer
from MT2D_secondary import *

def generate_model(alpha_l,n_sample,nza,n_freq,size_k,step,size_k_coarse):
    '''
    genenrate random electrical model  

    Three parts of model:
    0. layers (1D model) for background
    1. blocks
    2. layers
    3. (one) fault

    Parameters:
    nza   : number of air layer
    y     : [-y,y]
    z     : [0,z] 
    size_o: number of observation stations
    size_k: kernel domain,
    '''
    z = 100e3 
    y = 100e3
    # size_k = 100
    # n_freq = 50
    size_b = nza
    # freq = np.logspace(1,np.log10(1/200),n_freq)
    freq = np.logspace(1.1, np.log10(1/200),  n_freq)   # testing for different frequency

    freq_coarse = freq
    # notice!
    ry = np.linspace(-y,y,size_k)
    ry_coarse = ry#[::step][:size_k_coarse]
    multiple_t = 4.0
    multiple_b = 4.0
    multiple_l = 4.0
    multiple_r = 4.0
    z_air = -(np.logspace(0, np.log10(multiple_t*z), nza+1))[::-1]
    # near surface
    # zn0 = np.concatenate(([0],np.linspace(0.1, z, size_k)))

    #################### z in depth #######################
    # z1 = np.array([0, 2, 8, 20, 50, 120, 250, 400, 500, 700, 1000],dtype=float)
    z1 = np.array([0, 0.1, 1, 3, 6, 10, 16, 25, 40, 60, 80, \
                  100, 140, 190, 240, 300, 400, 500, 600, 800, 1000],dtype=float)
    n_z1 = len(z1)
    z2 = np.logspace(np.log10(1e3), np.log10(20e3), int((size_k-n_z1)/2))
    n_z2 = len(z2)
    z3 = np.logspace(np.log10(20e3),np.log10(z),size_k+3-n_z2-n_z1)
    # z4 = np.logspace(log10(100e3),log10(410e3),5)
    zn0 = np.concatenate((z1[:-1],z2[:-1],z3))
    zn0_coarse = zn0[::step][:size_k_coarse+1]
    #######################################################

    # zn0 = np.concatenate(([0],np.logspace(0.1, np.log10(z), size_k)))
    z_b = np.logspace(np.log10(zn0[-1]),np.log10(multiple_b*zn0[-1]),size_b+1)
    zn  = np.concatenate((z_air[:-1],zn0,z_b[1:]))
    zn_coarse = np.concatenate((z_air[:-1],zn0_coarse,z_b[1:]))

    yn0 = np.linspace(-y,y,size_k+1)
    yn0_coarse = yn0[::step][:size_k_coarse+1]
    # expand non kernel domain
    y_l = -(np.logspace(np.log10(multiple_l*yn0[-1]),np.log10(yn0[-1]),size_b+1))
    y_r = np.logspace(np.log10(yn0[-1]),np.log10(multiple_r*yn0[-1]),size_b+1)
    yn  = np.concatenate((y_l[:-1],yn0,y_r[1:]))
    yn_coarse = np.concatenate((y_l[:-1],yn0_coarse,y_r[1:]))

    ############## background and air layer #####################
    len_z = size_b+size_k+nza
    len_y = 2*size_b+size_k
    model = np.ones((n_sample,len_z,len_y))*(-2)# 100ohm.m for background
    model_coarse = np.ones((n_sample,size_b+size_k_coarse+nza,2*size_b+size_k_coarse))*(-2)# 100ohm.m for background
    #############  parameters  ###################################
    sig_up, sig_down = -4, 0  # sigma range from 10^[0,-5];

    # random
    # alpha = 4.0 # smothness of model, the larger the smoother
    mode = 'bound' # 
    set_1 = sig_up # log10
    set_2 = sig_down # log10
    # size_random = size_k#+size_b*2
    # blocks
    bound_block0, bound_block1 = 1, 5# number of blocks: 1~5 blocks
    top_z1,top_z2 = 1,5 # top of blocks, 1km~5km
    thick_b1, thick_b2 = 2,10 # thickness of blocks, 2~10km;

    ################  construct model ###############################
    for ii in range(n_sample):

        # random model
        model0 = 0.0
        model1 = 0.0
        # alpha_l = [3.0,4.0,5.0,6.0,7.0]
        # alpha_l = [3.0]
        for alpha in alpha_l:
        # 1. random conductivity for whold domain (not just kernel domain)
            model_temp0 = gr.gaussian_random_field(alpha = alpha, size = size_k,
                                       mode=mode, set_1=set_1, set_2=set_2)
            model_temp1 = gr.gaussian_random_field(alpha = alpha, size = size_k,
                                       mode=mode, set_1=-3, set_2=set_2) # for near surface
            model1 += model_temp1
            model0 += model_temp0
        min0 = np.min(model_temp0)
        max0 = np.max(model_temp0)
        # near surface
        model0[:int(size_k/5),:] = model1[:int(size_k/5),:]
        model0 = gaussian_filter(model0, sigma=2)/len(alpha_l)
        min1 = np.min(model0)
        max1 = np.max(model0)
        model0 = (model0-min1)*((max0-min0)/(max1-min1))+min0
        model[ii,nza:-size_b,size_b:-size_b] = model0
        model[ii,nza:-size_b,:size_b]  = model0[:,0:1]*np.ones((size_k,size_b)) # left
        model[ii,nza:-size_b,-size_b:] = model0[:,-1:]*np.ones((size_k,size_b))#right
        model[ii,-size_b:,:]           = -2 #bottom
        # bottom, interpolate from bottom to expanded bottom (100 ohm.m)
        sig_bottom = np.concatenate((model[ii,-size_b-1:-size_b,:],(-2)*np.ones_like(model[ii,-size_b-1:-size_b,:])),0).T
        f = interp2d(np.concatenate([zn0[-1:],[zn[int(-size_b/2)]]]),yn[1:], sig_bottom, kind='linear')
        model[ii,-size_b:int(-size_b/2),:]           = f(zn[-size_b:int(-size_b/2)],yn[1:]).T

        # blocks
        # the Code below is copied from one_blocks.py
        num_block  = np.random.randint(bound_block0, bound_block1,1)[0]
        y_idx0 = np.sort(random.sample(np.linspace(-y+1e3, y-1e3,1000).tolist(), num_block*2))
        # top of blocks start at z_idx km
        z_idx = np.random.randint(top_z1,    top_z2, num_block)*1e3
        # thickness of blocks
        height_idx = np.random.randint(thick_b1, thick_b2, num_block)*1e3
        # height_idx = np.random.randint(2, 15, num_block)*1e3
        # sigma
        sig_block  = np.random.uniform(sig_up, sig_down, num_block)
        for kk in range(num_block):
            for gg in range(size_k):
                for hh in range(size_k):
                    if y_idx0[2*kk]<=yn0[hh]<=y_idx0[2*kk+1] and z_idx[kk]<=zn0[gg]<=z_idx[kk]+height_idx[kk]:
                        model[ii,nza+gg:nza+gg+1,size_b+hh:size_b+hh+1] = sig_block[kk]

        # first layer, near surface, sedimentary:5-900 ohm.m
        sig_0  = np.random.uniform(-3,0,1)[0]
        model[ii,nza:nza+1,:] = sig_0 
        model_coarse[ii,nza:nza+1,:] = sig_0 

    # subsample from fine model to get coarse model
    model[:,:nza,:] = -9
    model_coarse[:,:nza,:] = -9 # top
    model_coarse[:,-size_b:,size_b:-size_b] = model[:,-size_b:,size_b:-size_b][::1,::1,::step][:,:,:size_k_coarse] # bottom
    model_coarse[:,-size_b:,:size_b] = model[:,-size_b:,:size_b] # left bottom
    model_coarse[:,-size_b:,-size_b:]= model[:,-size_b:,-size_b:] # right bottom
    model_coarse[:,nza:-size_b,:size_b]  = model[:,nza:-size_b,:size_b][::1,::step,::1][:,:size_k_coarse,:] # left 
    model_coarse[:,nza:-size_b,-size_b:] = model[:,nza:-size_b,-size_b:][::1,::step,::1][:,:size_k_coarse,:] # right
    model = 10**model 
    model_coarse = 10**model_coarse 
    model_k = model[:,nza:-size_b,size_b:-size_b]
    model_k_coarse = model_k[:,::step,::step][:,:size_k_coarse,:size_k_coarse]
    model_coarse[:,nza:-size_b,size_b:-size_b] =  model_k_coarse

    return zn, yn, freq, ry, model,zn0,yn0,model_k,\
        zn_coarse, yn_coarse, freq_coarse, ry_coarse, model_coarse,zn0_coarse,yn0_coarse,model_k_coarse

def inter_model(self,y0,z0,sig0,y,z):
    '''
    not finished!

    linear interpolation of model. 
    Use refined grid for forward modeling while coarse grid for train.

    Parameters: 
    y0: y coordinates of refined grid
    z0: z coordinates of refined grid
    sig0: sigma in refined grid
    y: y coordinates of coarse grid
    z: z coordinates of coarse grid
    '''
    # f = interpolate.interp2d(x, y, z, kind='linear')
    # sig = f(y,z)
    # return sig
    pass


def save_model(model_name0,zn, yn, freq, ry, sig_log, rhoxy, phsxy,zxy,rhoyx,phsyx,zyx):
    '''
    save data as electrical model and field 
    for field, save as matrix with size of (n_model, n_obs, n_freq)

    '''
    model_name = model_name0+'.mat'
    scio.savemat(model_name,{'zn':zn, 'yn':yn, 'freq':freq, 'obs':ry,'sig':sig_log,
                            'rhoxy':rhoxy, 'phsxy':phsxy,'zxy':zxy,
                            'rhoyx':rhoyx,'phsyx':phsyx,'zyx':zyx})

def func_remote(nza, zn, yn, freq, ry, sig,n_sample,mode="TETM",np_dtype = np.float64):
    n_freq = np.size(freq)
    n_ry   = len(ry)
    rhoxy = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype) 
    phsxy = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    rhoyx = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    phsyx = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    zxy   = np.zeros((n_sample,n_freq,n_ry),dtype=complex)
    zyx   = np.zeros((n_sample,n_freq,n_ry),dtype=complex)

    # rhoxy, phsxy,Zxy,rhoyx,phsyx,Zyx  = model.mt2d("TETM")
    result = []
    for ii in range(n_sample):
        model = MT2DFD.remote(nza, zn, yn, freq, ry, sig[ii,:,:])
        result.append(model.mt2d.remote(mode))

    temp0 = ray.get(result)
    for ii in range(len(temp0)):
        temp = temp0[ii]
        # log10(rho)
        rhoxy[ii,:,:], phsxy[ii,:,:],zxy[ii,:,:],rhoyx[ii,:,:],phsyx[ii,:,:],zyx[ii,:,:]  =\
            temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]
    
    # print("remote computation finished !")
    return rhoxy, phsxy,zxy,rhoyx,phsyx,zyx

def main(n_sample,num_cpus,model_name):
    np_dtype = np.float64
    nza = 10 # number of air layer
    n_freq = 64 # number of frequency
    size_k = 64
    # n_freq_coarse = n_freq # number of frequency
    step = 2
    size_k_coarse = int(size_k/step)
    alpha_l = [3.0,4.0,5.0,6.0,7.0]


    zn, yn, freq, ry, sig,zn0,yn0,sig_k,\
        zn_coarse, yn_coarse, freq_coarse, ry_coarse, sig_coarse,zn0_coarse,yn0_coarse,sig_k_coarse\
             = generate_model(alpha_l,n_sample,nza,n_freq,size_k,step,size_k_coarse)

    time0 = default_timer()

    ray.init()


    # skin depth 
    alpha0 = 1 # 1 times of skin depth
    skin_depth = 0.503*alpha0*np.sqrt(1.0/np.mean(sig_k)/freq[0]) # km
    print(f"skin depth is {skin_depth}km")
    skin_depth = 0.503*alpha0*np.sqrt(1.0/np.mean(sig_k)/freq[-1]) # km
    print(f"skin depth is {skin_depth}km")


    k = int(n_sample/num_cpus)
    n_sample = int(k*num_cpus)
    n_ry = len(ry)
    rhoxy = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype) 
    phsxy = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    rhoyx = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    phsyx = np.zeros((n_sample,n_freq,n_ry),dtype=np_dtype)
    zxy   = np.zeros((n_sample,n_freq,n_ry),dtype=complex)
    zyx   = np.zeros((n_sample,n_freq,n_ry),dtype=complex)
    for ii in range(k):
        rhoxy[num_cpus*ii:num_cpus*(ii+1),:,:], phsxy[num_cpus*ii:num_cpus*(ii+1),:,:],zxy[num_cpus*ii:num_cpus*(ii+1),:,:],\
            rhoyx[num_cpus*ii:num_cpus*(ii+1),:,:],phsyx[num_cpus*ii:num_cpus*(ii+1),:,:],zyx[num_cpus*ii:num_cpus*(ii+1),:,:]  = \
            func_remote(nza, zn, yn, freq, ry, sig[num_cpus*ii:num_cpus*(ii+1),:,:],n_sample=num_cpus,mode="TETM")
        print(f"{ii} of {k} finished!")

    rhoxy = np.log10(rhoxy)
    rhoyx = np.log10(rhoyx)
    # notice: keep the same number of sig and rho,phs
    sig_k = np.log10(sig_k[:n_sample,:,:])

    save_model(model_name,zn0, yn0, freq, ry, sig_k, rhoxy, phsxy,zxy,rhoyx,phsyx,zyx)

    time1 = default_timer()
    # rhoxy, phsxy,Zxy, rhoyx, phsyx,Zyx  = model.mt2d("TETM")
    # save_model(filename,resis, rhoxy, phsxy, rhoyx, phsyx)
    print(f"time using: {time1-time0}s")

    # print("calculate coarse grid")

    # time0 = default_timer()
    # model_name_coarse = model_name+'_coarse'
    # n_ry_coarse = len(ry_coarse)
    # rhoxy_coarse = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=np_dtype) 
    # phsxy_coarse = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=np_dtype)
    # rhoyx_coarse = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=np_dtype)
    # phsyx_coarse = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=np_dtype)
    # zxy_coarse   = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=complex)
    # zyx_coarse   = np.zeros((n_sample,n_freq_coarse,n_ry_coarse),dtype=complex)
    # for ii in range(k):
    #     rhoxy_coarse[num_cpus*ii:num_cpus*(ii+1),:,:], phsxy_coarse[num_cpus*ii:num_cpus*(ii+1),:,:],zxy_coarse[num_cpus*ii:num_cpus*(ii+1),:,:],\
    #         rhoyx_coarse[num_cpus*ii:num_cpus*(ii+1),:,:],phsyx_coarse[num_cpus*ii:num_cpus*(ii+1),:,:],zyx_coarse[num_cpus*ii:num_cpus*(ii+1),:,:]  = \
    #         func_remote(nza, zn_coarse, yn_coarse, freq_coarse, ry_coarse, sig_coarse[num_cpus*ii:num_cpus*(ii+1),:,:],n_sample=num_cpus,mode="TETM")
    #     print(f"{ii} of {k} finished!")

    # rhoxy_coarse = np.log10(rhoxy_coarse)
    # rhoyx_coarse = np.log10(rhoyx_coarse)
    # # notice: keep the same number of sig and rho,phs
    # sig_k_coarse = np.log10(sig_k_coarse[:n_sample,:,:])

    # save_model(model_name_coarse,zn0_coarse, yn0_coarse, freq_coarse, ry_coarse, sig_k_coarse, \
    # rhoxy_coarse, phsxy_coarse,zxy_coarse,rhoyx_coarse,phsyx_coarse,zyx_coarse)

    # time1 = default_timer()
    # # rhoxy, phsxy,Zxy, rhoyx, phsyx,Zyx  = model.mt2d("TETM")
    # # save_model(filename,resis, rhoxy, phsxy, rhoyx, phsyx)
    # print(f"time using: {time1-time0}s")
    ray.shutdown()
 
if __name__ == "__main__":
    try:# run in command line
        n_sample = int(sys.argv[1]) # number of random model
        num_cpus = int(sys.argv[2])
        model_name = '../data/'+sys.argv[3]
    except:# debug in vscode
        print("debug mode, no input")
        n_sample = 1 # number of random model
        num_cpus = 1
        model_name = '../data/block'
    main(n_sample,num_cpus,model_name)