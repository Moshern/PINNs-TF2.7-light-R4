# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from pinns_2d import NS2D_UnSteady_PINNs
from pinns_3d import NS3D_UnSteady_PINNs
from funcs import flow2D



def predict_2d2c_liuyi_fsi():
    
    data_pathname = './data/liuyi'
    nlevels = [0,1,2,4,8,16]
    nlevels = [0]
    reslus = [5,10,20,40,80]
    reslus = [100]
    rnum = [0]
    for nlev in nlevels:
        for nres in reslus:
            data_filename = 'LiuYi_fish2D_reslu{nres}_noise{nlev}_uniform500.mat'.format(nres=nres,nlev=nlev)
            tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
            pred_xmesh = np.transpose(tmp['pred_xmesh'])
            pred_ymesh = np.transpose(tmp['pred_ymesh'])
            pred_xe = np.transpose(tmp['pred_xe'])
            pred_ye = np.transpose(tmp['pred_ye'])
            mint = np.transpose(tmp['mint'])
            maxt = np.transpose(tmp['maxt'])
            tvec = np.linspace(mint, maxt, 500)
            sizU = np.shape(pred_xmesh) + (len(tvec),)
            sizE = np.shape(pred_xe)     
            
            for runidx in rnum:
                savename = 'LiuYi_fish2D_13_160_9_48_reslu{nres}_noise{nlev}_uniform500_WangDyn_run{runidx}'.format(nres=nres,nlev=nlev,runidx=runidx)
                save_file = './weights/'+savename+'/'+savename
                domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
                norm_paras = domain['norm_paras']  
                        
                """
                parameters
                """       
                hp = {'layers':[3] + 13*[160] + [3],
                      'ExistModel':1,
                      'train':False,
                      'maptype':'rnn',
                      'savename':savename,
                      'Re':5000,
                      'alpha':1.0,
                      'norm_paras':norm_paras,
                      'tf_epochs':1000,
                      'tf_batch_size':10000,
                      'initial_epoch':0,
                      'init_lr':5.0e-3,
                      'bfgs_epochs':0,
                      'bfgs_batch_size':10000,
                      'lm_epochs':0,
                      'lm_batch_size':200}
                 
                # Load trained neural network
                pinn_model = NS2D_UnSteady_PINNs(hp)
                
                x_pred = pred_xmesh.flatten()[:,None]
                y_pred = pred_ymesh.flatten()[:,None]
            
            
                all_data_u = np.zeros((sizU[0],sizU[1],sizU[2]))
                all_data_v = np.zeros((sizU[0],sizU[1],sizU[2]))
                all_data_p = np.zeros((sizU[0],sizU[1],sizU[2]))
                all_data_e1 = np.zeros((sizU[0],sizU[1],sizU[2]))
                all_data_e2 = np.zeros((sizU[0],sizU[1],sizU[2]))
                all_data_e3 = np.zeros((sizU[0],sizU[1],sizU[2]))
                # x,y,z,ub,vb,wb,pb,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz
                edge = np.zeros((sizE[0],sizU[2],11))
                
                count = -1;
                for tt in tvec:
                    count = count+1
                    t_pred = tt*np.ones_like(x_pred)
                    pred = np.concatenate((t_pred,x_pred,y_pred), 1)
             
                    if np.mod(count,10) == 0:
                        print("--- loop %d in total %d ---" % (count, sizU[2]))
                        
                    # prediction
                    u_pred, v_pred, p_pred = pinn_model.predict_field(pred)
                    pred = pred.astype('float32')
                    e1, e2, e3 = pinn_model.ns_eqns(pred)
                    e1 = e1.numpy()
                    e2 = e2.numpy()
                    e3 = e3.numpy()
            
            
                    tmp = u_pred.reshape(sizU[0],sizU[1])
                    all_data_u[:,:,count] = tmp
                    tmp = v_pred.reshape(sizU[0],sizU[1])
                    all_data_v[:,:,count] = tmp
                    tmp = p_pred.reshape(sizU[0],sizU[1])
                    all_data_p[:,:,count] = tmp
                    
                    tmp = e1.reshape(sizU[0],sizU[1])
                    all_data_e1[:,:,count] = tmp
                    tmp = e2.reshape(sizU[0],sizU[1])
                    all_data_e2[:,:,count] = tmp
                    tmp = e3.reshape(sizU[0],sizU[1])
                    all_data_e3[:,:,count] = tmp
            
                # save the predicted data
                filepath = './predict_results'
                filename = ('%s_predict.mat') % (hp['savename'])
                sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
                                                   'all_data_u':all_data_u,\
                                                   'all_data_v':all_data_v,\
                                                   'all_data_p':all_data_p,\
                                                   'all_data_e1':all_data_e1,\
                                                   'all_data_e2':all_data_e2,\
                                                   'all_data_e3':all_data_e3})

                    
                    
def predict_3d3c_wanglei():
    
    data_pathname = './data/wanglei'
    data_filename = '3d3c_wanglei_noorifice_1_120.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    pred_xmesh = np.transpose(tmp['pred_xmesh'])
    pred_ymesh = np.transpose(tmp['pred_ymesh'])
    pred_zmesh = np.transpose(tmp['pred_zmesh'])
    mint = np.transpose(tmp['mint'])
    maxt = np.transpose(tmp['maxt'])
    tvec = np.linspace(mint, maxt, 120)
    sizU = np.shape(pred_xmesh) + (len(tvec),)
    

    savename = 'wanglei_noorifice_3d3c_15_256_1_120'
    save_file = './weights/'+savename
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    norm_paras = domain['norm_paras']  
            
    """
    parameters
    """       
    hp = {'layers':[4] + 15*[256] + [4],
          'ExistModel':1,
          'train':False,
          'maptype':'rnn',
          'savename':savename,
          'Re':19656,
          'alpha':1.0,
          'norm_paras':norm_paras,
          'tf_epochs':10000,
          'tf_batch_size':10000,
          'initial_epoch':0,
          'init_lr':5.0e-3,
          'bfgs_epochs':0,
          'bfgs_batch_size':10000,
          'lm_epochs':0,
          'lm_batch_size':50}
     
    # Load trained neural network
    pinn_model = NS3D_UnSteady_PINNs(hp)
    
    x_pred = pred_xmesh.flatten()[:,None]
    y_pred = pred_ymesh.flatten()[:,None]
    z_pred = pred_zmesh.flatten()[:,None]


    all_data_u = np.zeros((sizU[0],sizU[1],sizU[2],sizU[3]))
    all_data_v = np.zeros((sizU[0],sizU[1],sizU[2],sizU[3]))
    all_data_w = np.zeros((sizU[0],sizU[1],sizU[2],sizU[3]))
    all_data_p = np.zeros((sizU[0],sizU[1],sizU[2],sizU[3]))

    
    count = -1;
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred,z_pred), 1)
 
        # prediction
        u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred)
        pred = pred.astype('float32')
  
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizU[3]))

        tmp = u_pred.reshape(sizU[0],sizU[1],sizU[2])
        all_data_u[:,:,:,count] = tmp
        tmp = v_pred.reshape(sizU[0],sizU[1],sizU[2])
        all_data_v[:,:,:,count] = tmp
        tmp = w_pred.reshape(sizU[0],sizU[1],sizU[2])
        all_data_w[:,:,:,count] = tmp
        tmp = p_pred.reshape(sizU[0],sizU[1],sizU[2])
        all_data_p[:,:,:,count] = tmp
        
    

    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh, 'zmesh':pred_zmesh,\
                                       'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v,\
                                       'all_data_w':all_data_w,\
                                       'all_data_p':all_data_p})





def predict_2d2c_zangzy():
    
    data_pathname = './data/zangzy'
    data_filename = 'vortexrings-leapfrogging_zangzy_traindata.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    pred_xmesh = np.transpose(tmp['pred_xmesh'])
    pred_ymesh = np.transpose(tmp['pred_ymesh'])
    mint = np.transpose(tmp['mint'])
    maxt = np.transpose(tmp['maxt'])
    tvec = np.linspace(mint, maxt, 200)
    sizU = np.shape(pred_xmesh) + (len(tvec),)
    

    savename = 'vortexrings-leapfrogging_zangzy_U1L1'
    save_file = './weights/'+savename+'/'+savename
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    norm_paras = domain['norm_paras']  
            
    """
    parameters
    """       
    hp = {'layers':[3] + 11*[128] + [3],
          'ExistModel':1,
          'train':False,
          'maptype':'rnn',
          'savename':savename,
          'Re':2500.0,
          'alpha':1.0,
          'norm_paras':norm_paras,
          'tf_epochs':10000,
          'tf_batch_size':10000,
          'initial_epoch':0,
          'init_lr':5.0e-3,
          'bfgs_epochs':0,
          'bfgs_batch_size':10000,
          'lm_epochs':0,
          'lm_batch_size':50}
     
    # Load trained neural network
    pinn_model = NS2D_UnSteady_PINNs(hp)
    
    x_pred = pred_xmesh.flatten()[:,None]
    y_pred = pred_ymesh.flatten()[:,None]


    all_data_u = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_v = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_p = np.zeros((sizU[0],sizU[1],sizU[2]))

    
    count = -1;
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred), 1)
 
        # prediction
        u_pred, v_pred, p_pred = pinn_model.predict_field(pred)
        pred = pred.astype('float32')
  
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizU[2]))

        tmp = u_pred.reshape(sizU[0],sizU[1])
        all_data_u[:,:,count] = tmp
        tmp = v_pred.reshape(sizU[0],sizU[1])
        all_data_v[:,:,count] = tmp
        tmp = p_pred.reshape(sizU[0],sizU[1])
        all_data_p[:,:,count] = tmp
        
    

    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
                                       'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v,\
                                       'all_data_p':all_data_p})
                    
  




def predict_2d2c_wufan():
    
    data_pathname = './data/wufan'
    data_filename = 'wufan_movingcylinder2D_reslu40_noise8_pinn.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    pred_xmesh = np.transpose(tmp['pred_xmesh'])
    pred_ymesh = np.transpose(tmp['pred_ymesh'])
    mint = np.transpose(tmp['mint'])
    maxt = np.transpose(tmp['maxt'])
    tvec = np.linspace(mint, maxt, 400)
    sizU = np.shape(pred_xmesh) + (len(tvec),)
    

    savename = 'wufan_movingcylinder2D_13_128_reslu40_noise8_pinn_run0'
    save_file = './weights/'+savename+'/'+savename
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    norm_paras = domain['norm_paras']  
            
    """
    parameters
    """       
    hp = {'layers':[3] + 13*[128] + [3],
          'ExistModel':1,
          'train':False,
          'maptype':'rnn',
          'savename':savename,
          'Re':185.0,
          'alpha':1.0,
          'norm_paras':norm_paras,
          'tf_epochs':4000,
          'tf_batch_size':10000,
          'initial_epoch':0,
          'init_lr':5.0e-3,
          'bfgs_epochs':0,
          'bfgs_batch_size':10000,
          'lm_epochs':0,
          'lm_batch_size':50}
     
    # Load trained neural network
    pinn_model = NS2D_UnSteady_PINNs(hp)
    
    x_pred = pred_xmesh.flatten()[:,None]
    y_pred = pred_ymesh.flatten()[:,None]


    all_data_u = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_v = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_p = np.zeros((sizU[0],sizU[1],sizU[2]))

    
    count = -1;
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred), 1)
 
        # prediction
        u_pred, v_pred, p_pred = pinn_model.predict_field(pred)
        pred = pred.astype('float32')
  
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizU[2]))

        tmp = u_pred.reshape(sizU[0],sizU[1])
        all_data_u[:,:,count] = tmp
        tmp = v_pred.reshape(sizU[0],sizU[1])
        all_data_v[:,:,count] = tmp
        tmp = p_pred.reshape(sizU[0],sizU[1])
        all_data_p[:,:,count] = tmp
        
    

    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
                                       'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v,\
                                       'all_data_p':all_data_p})


def predict_2d2c_bendpipe():
    
    # 修改自predict_2d2c_wufan()并且尝试输出方程残差
    
    data_pathname = './data/bendpipe'
    data_filename = '2dWo10Bpi32_reslu40_noise0_pinn.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    pred_xmesh = np.transpose(tmp['pred_xmesh'])
    pred_ymesh = np.transpose(tmp['pred_ymesh'])
    mint = np.transpose(tmp['mint'])
    maxt = np.transpose(tmp['maxt'])
    tvec = np.linspace(mint, maxt, 201)
    sizU = np.shape(pred_xmesh) + (len(tvec),)
    

    savename = '2dWo10Bpi32_13_128'
    save_file = './weights/'+savename+'/'+savename
    domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
    norm_paras = domain['norm_paras']  
            
    hp = {'layers':[3] + 13*[128] + [3],
          'ExistModel':1,
          'train':False,
          'maptype':'rnn',
          'savename':savename,
          'Re':1000.0,
          'alpha':1.0,
          'norm_paras':norm_paras,
          'tf_epochs':6000,
          'tf_batch_size':10000,
          'initial_epoch':0,
          'init_lr':1.0e-3,
          'bfgs_epochs':0,
          'bfgs_batch_size':10000,
          'lm_epochs':0,
          'lm_batch_size':50}
     
    # Load trained neural network
    pinn_model = NS2D_UnSteady_PINNs(hp)
    
    x_pred = pred_xmesh.flatten()[:,None]
    y_pred = pred_ymesh.flatten()[:,None]


    all_data_u = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_v = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_p = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_e1 = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_e2 = np.zeros((sizU[0],sizU[1],sizU[2]))
    all_data_e3 = np.zeros((sizU[0],sizU[1],sizU[2]))
    # x,y,z,ub,vb,wb,pb,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz
    # edge = np.zeros((sizE[0],sizU[2],11))
    
    count = -1;
    for tt in tvec:
        count = count+1
        t_pred = tt*np.ones_like(x_pred)
        pred = np.concatenate((t_pred,x_pred,y_pred), 1)
 
        if np.mod(count,10) == 0:
            print("--- loop %d in total %d ---" % (count, sizU[2]))
            
        # prediction
        u_pred, v_pred, p_pred = pinn_model.predict_field(pred)
        print('over')
        pred = pred.astype('float32')
        e1, e2, e3 = pinn_model.ns_eqns(pred)
        e1 = e1.numpy()
        e2 = e2.numpy()
        e3 = e3.numpy()


        tmp = u_pred.reshape(sizU[0],sizU[1])
        all_data_u[:,:,count] = tmp
        tmp = v_pred.reshape(sizU[0],sizU[1])
        all_data_v[:,:,count] = tmp
        tmp = p_pred.reshape(sizU[0],sizU[1])
        all_data_p[:,:,count] = tmp
        
        tmp = e1.reshape(sizU[0],sizU[1])
        all_data_e1[:,:,count] = tmp
        tmp = e2.reshape(sizU[0],sizU[1])
        all_data_e2[:,:,count] = tmp
        tmp = e3.reshape(sizU[0],sizU[1])
        all_data_e3[:,:,count] = tmp

    # save the predicted data
    filepath = './predict_results'
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
                                       'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v,\
                                       'all_data_p':all_data_p,\
                                       'all_data_e1':all_data_e1,\
                                       'all_data_e2':all_data_e2,\
                                       'all_data_e3':all_data_e3})
        


# def predict_2d2c_bendpipe():
    
#     # 修改自predict_2d2c_wufan()
    
#     data_pathname = './data/bendpipe'
#     data_filename = '2PRe392p5wpi_16_reslu40_noise0_pinn.mat'
#     tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
#     pred_xmesh = np.transpose(tmp['pred_xmesh'])
#     pred_ymesh = np.transpose(tmp['pred_ymesh'])
#     mint = np.transpose(tmp['mint'])
#     maxt = np.transpose(tmp['maxt'])
#     tvec = np.linspace(mint, maxt, 321)
#     sizU = np.shape(pred_xmesh) + (len(tvec),)
    

#     savename = '2PRe392p5wpi_16_2d2c_13_128'
#     save_file = './weights/'+savename+'/'+savename
#     domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
#     norm_paras = domain['norm_paras']  
            
#     hp = {'layers':[3] + 13*[128] + [3],
#           'ExistModel':1,
#           'train':False,
#           'maptype':'rnn',
#           'savename':savename,
#           'Re':392.5,
#           'alpha':1.0,
#           'norm_paras':norm_paras,
#           'tf_epochs':4000,
#           'tf_batch_size':10000,
#           'initial_epoch':0,
#           'init_lr':1.0e-3,
#           'bfgs_epochs':0,
#           'bfgs_batch_size':10000,
#           'lm_epochs':0,
#           'lm_batch_size':50}
     
#     # Load trained neural network
#     pinn_model = NS2D_UnSteady_PINNs(hp)
    
#     x_pred = pred_xmesh.flatten()[:,None]
#     y_pred = pred_ymesh.flatten()[:,None]


#     all_data_u = np.zeros((sizU[0],sizU[1],sizU[2]))
#     all_data_v = np.zeros((sizU[0],sizU[1],sizU[2]))
#     all_data_p = np.zeros((sizU[0],sizU[1],sizU[2]))

    
#     count = -1;
#     for tt in tvec:
#         count = count+1
#         t_pred = tt*np.ones_like(x_pred)
#         pred = np.concatenate((t_pred,x_pred,y_pred), 1)
 
#         # prediction
#         u_pred, v_pred, p_pred = pinn_model.predict_field(pred)
#         pred = pred.astype('float32')
  
#         if np.mod(count,10) == 0:
#             print("--- loop %d in total %d ---" % (count, sizU[2]))

#         tmp = u_pred.reshape(sizU[0],sizU[1])
#         all_data_u[:,:,count] = tmp
#         tmp = v_pred.reshape(sizU[0],sizU[1])
#         all_data_v[:,:,count] = tmp
#         tmp = p_pred.reshape(sizU[0],sizU[1])
#         all_data_p[:,:,count] = tmp
        
    

#     # save the predicted data
#     filepath = './predict_results'
#     filename = ('%s_predict.mat') % (hp['savename'])
#     sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
#                                        'all_data_u':all_data_u,\
#                                        'all_data_v':all_data_v,\
#                                        'all_data_p':all_data_p})
        
    
# def predict_3d3c_AOCFD():
    
#     data_pathname = './data/bendpipe'
#     data_filename = '2d3c_2PInRe785wpi_4_reslu40_noise0_pinn.mat'
#     tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
#     pred_xmesh = np.transpose(tmp['pred_xmesh'])
#     pred_ymesh = np.transpose(tmp['pred_ymesh'])
#     pred_zmesh = np.transpose(tmp['pred_zmesh'])
#     pred_xe = np.transpose(tmp['pred_xe'])
#     pred_ye = np.transpose(tmp['pred_ye'])
#     # pred_ze = np.transpose(tmp['pred_ze'])
#     mint = np.transpose(tmp['mint'])
#     maxt = np.transpose(tmp['maxt'])
#     tvec = np.linspace(mint, maxt, 161)
#     sizU = np.shape(pred_xmesh)
#     sizE = np.shape(pred_xe)
#     print('pred_xmesh/sizU:', pred_xmesh.shape)
#     print('pred_xe/sizE:', pred_xe.shape)
    
    

#     savename = '2d3c_2PInRe785wpi_4_run0'
#     save_file = './weights/'+savename+'/'+savename
#     domain = sio.loadmat(save_file+'_paras.mat',squeeze_me=True)
#     norm_paras = domain['norm_paras']  

            
#     """
#     parameters
#     """       
#     hp = {'layers':[4] + 13*[128] + [4],
#           'ExistModel':1,
#           'train':False,
#           'maptype':'rnn',
#           'savename':savename,
#           'Re':785,
#           'alpha':1.0,
#           'norm_paras':norm_paras,
#           'tf_epochs':2000,
#           'tf_batch_size':8000,
#           'initial_epoch':0,
#           'init_lr':1.0e-3,        
#           'bfgs_epochs':0,
#           'bfgs_batch_size':10000,
#           'lm_epochs':0,
#           'lm_batch_size':200}
     
#     # Load trained neural network
#     pinn_model = NS3D_UnSteady_PINNs(hp)
    
#     x_pred = pred_xmesh[:, 0]
#     y_pred = pred_ymesh[:, 0]
#     z_pred = pred_zmesh[:, 0]
#     x_pred = np.expand_dims(x_pred, axis=1)
#     y_pred = np.expand_dims(y_pred, axis=1)
#     z_pred = np.expand_dims(z_pred, axis=1)

#     all_data_u = np.zeros((sizU[0],sizU[1]))
#     all_data_v = np.zeros((sizU[0],sizU[1]))
#     all_data_w = np.zeros((sizU[0],sizU[1]))
#     all_data_p = np.zeros((sizU[0],sizU[1]))
#     all_data_e1 = np.zeros((sizU[0],sizU[1]))
#     all_data_e2 = np.zeros((sizU[0],sizU[1]))
#     all_data_e3 = np.zeros((sizU[0],sizU[1]))
#     all_data_e4 = np.zeros((sizU[0],sizU[1]))

#     # x,y,z,ub,vb,wb,pb,dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz
#     edge = np.zeros((sizE[0], sizU[1], 19))

    
#     count = -1;
#     for tt in tvec:
#         count = count+1
#         t_pred = tt*np.ones_like(x_pred)
        
#         print('x_pred:', x_pred.shape)
#         print('y_pred:', y_pred.shape)
#         print('z_pred:', z_pred.shape)
#         print('t_pred:', t_pred.shape)
#         pred = np.concatenate((t_pred,x_pred,y_pred,z_pred), 1)
#         print('pred:', pred.shape)
 
#         if np.mod(count,10) == 0:
#             print("--- loop %d in total %d ---" % (count, sizU[1]))
            
#         # prediction
#         u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred)
#         print('over')
#         pred = pred.astype('float32')
#         e1, e2, e3, e4 = pinn_model.ns_eqns(pred)
#         e1 = e1.numpy()
#         e2 = e2.numpy()
#         e3 = e3.numpy()
#         e4 = e4.numpy()

#         tmp = u_pred.reshape(sizU[0])
#         all_data_u[:, count] = tmp
#         tmp = v_pred.reshape(sizU[0])
#         all_data_v[:, count] = tmp
#         tmp = w_pred.reshape(sizU[0])
#         all_data_w[:, count] = tmp        
#         tmp = p_pred.reshape(sizU[0])
#         all_data_p[:, count] = tmp
        
#         tmp = e1.reshape(sizU[0])
#         all_data_e1[:,count] = tmp
#         tmp = e2.reshape(sizU[0])
#         all_data_e2[:,count] = tmp
#         tmp = e3.reshape(sizU[0])
#         all_data_e3[:,count] = tmp
#         tmp = e4.reshape(sizU[0])
#         all_data_e4[:,count] = tmp
        
#         t_pred = tt * np.ones_like(pred_xe)
#         # pred = np.concatenate((t_pred, pred_xe, pred_ye, pred_ze), 1)
#         pred = np.concatenate((t_pred, pred_xe, pred_ye), 1)

#         # prediction
#         # xb, yb, zb, ub, vb, wb, pb, grad = pinn_model.predict_boundary(pred)
  
#         # if np.mod(count,10) == 0:
#         #     print("--- loop %d in total %d ---" % (count, sizU[1]))
#         # # print('xb:', xb.shape)
#         # edge[:,count,0:1] = xb
#         # edge[:,count,1:2] = yb       
#         # edge[:,count,2:3] = zb
#         # edge[:,count,3:4] = ub         
#         # edge[:,count,4:5] = vb    
#         # edge[:,count,5:6] = wb   
#         # edge[:,count,6:7] = pb  
#         # edge[:,count,7:8] = grad['ux']   
#         # edge[:,count,8:9] = grad['uy']   
#         # edge[:,count,9:10] = grad['uz']  
#         # edge[:,count,10:11] = grad['vx']  
#         # edge[:,count,11:12] = grad['vy']  
#         # edge[:,count,12:13] = grad['vz']  
#         # edge[:,count,13:14] = grad['wx'] 
#         # edge[:,count,14:15] = grad['wy']  
#         # edge[:,count,15:16] = grad['wz']  
#         # edge[:,count,16:17] = grad['px']  
#         # edge[:,count,17:18] = grad['py']  
#         # edge[:,count,18:19] = grad['pz']  
        
#     # save the predicted data
#     filepath = './predict_results'
#     filename = ('%s_predict.mat') % (hp['savename'])

#     pred_xmesh = pred_xmesh[:, 0]
#     pred_ymesh = pred_ymesh[:, 0]
#     pred_zmesh = pred_zmesh[:, 0]
#     pred_xmesh = pred_xmesh.T
#     pred_ymesh = pred_ymesh.T
#     pred_zmesh = pred_zmesh.T
#     #all_data_u = all_data_u.T
#     #all_data_v = all_data_v.T
#     #all_data_w = all_data_w.T
#     #all_data_p = all_data_p.T
#     print('pred_xmesh:', pred_xmesh.shape)
#     print('pred_ymesh:', pred_ymesh.shape)
#     print('pred_zmesh:', pred_zmesh.shape)
#     print('all_data_u:', all_data_u.shape)
#     print('all_data_v:', all_data_v.shape)
#     print('all_data_w:', all_data_w.shape)
#     print('all_data_p:', all_data_p.shape)
#     # print('edge:', edge.shape)

#     sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,'zmesh':pred_zmesh,\
#                                        'all_data_u':all_data_u,\
#                                        'all_data_v':all_data_v,\
#                                        'all_data_w':all_data_w,\
#                                        'all_data_p':all_data_p,\
#                                        'all_data_e1':all_data_e1,\
#                                        'all_data_e2':all_data_e2,\
#                                        'all_data_e3':all_data_e3,\
#                                        'all_data_e4':all_data_e4})


# def predict_3d3c_AOCFD():
#     """
#     这个是原来的3D文件只预测纵截面能使用的代码
#     Reduced-memory version of predict_3d3c_AOCFD:
#     - Only predicts on z = 0 plane
#     - Removes e1–e4 residual prediction
#     """

#     # ==== 加载网格 ====
#     data_pathname = './data/bendpipe'
#     data_filename = 'inlet_Wo10Bpi32_reslu40_noise0_pinn.mat'
#     tmp = h5py.File(os.path.join(data_pathname, data_filename), 'r')
#     pred_xmesh = np.transpose(tmp['pred_xmesh'])
#     pred_ymesh = np.transpose(tmp['pred_ymesh'])
#     pred_zmesh = np.transpose(tmp['pred_zmesh'])
#     mint = np.transpose(tmp['mint'])
#     maxt = np.transpose(tmp['maxt'])
#     tmp.close()

#     # # ==== 仅取 z=0 平面 ====
#     # pred_xmesh_2d = pred_xmesh[:, :, 0]
#     # pred_ymesh_2d = pred_ymesh[:, :, 0]
#     # z0 = pred_zmesh[0, 0, 0]   # 通常为 0，可直接写 z0 = 0

#     # # 3d3c时使用，找到 z 方向上最接近 0 的层索引
#     # z_levels = pred_zmesh[0, 0, :]   # 一定要有 :
#     # idx_z0 = np.argmin(np.abs(z_levels))
#     # z0 = float(z_levels[idx_z0])
    
#     # pred_xmesh_2d = pred_xmesh[:, :, idx_z0]
#     # pred_ymesh_2d = pred_ymesh[:, :, idx_z0]

#     # ==== 仅取 z=0 平面（2D 网格）====
#     pred_xmesh_2d = pred_xmesh
#     pred_ymesh_2d = pred_ymesh
    
#     # z 是常数（从数据里取，或者直接设 0）
#     z0 = float(pred_zmesh[0, 0])

#     mint = float(mint)
#     maxt = float(maxt)
#     tvec = np.linspace(mint, maxt, 201)
#     sizU = pred_xmesh_2d.shape + (len(tvec),)

#     # ==== 加载模型 ====
#     savename = 'inlet_Wo10Bpi32_13_156_run0'
#     save_file = './weights/' + savename + '/' + savename
#     domain = sio.loadmat(save_file + '_paras.mat', squeeze_me=True)
#     norm_paras = domain['norm_paras']

#     hp = {
#         'layers': [4] + 13 * [156] + [4],
#         'ExistModel': 1,
#         'train': False,
#         'maptype': 'rnn',
#         'savename': savename,
#         'Re': 1000.0,
#         'alpha': 1.0,
#         'norm_paras': norm_paras,
#         'tf_epochs': 8000,
#         'tf_batch_size': 5000,
#         'initial_epoch': 0,
#         'init_lr': 1.0e-3,
#         'bfgs_epochs': 0,
#         'bfgs_batch_size': 10000,
#         'lm_epochs': 0,
#         'lm_batch_size': 200
#     }

#     pinn_model = NS3D_UnSteady_PINNs(hp)

#     # ==== 网格展开 ====
#     x_pred = pred_xmesh_2d.flatten()[:, None]
#     y_pred = pred_ymesh_2d.flatten()[:, None]
#     z_pred = z0 * np.ones_like(x_pred)

#     # ==== 初始化预测数据 ====
#     all_data_u = np.zeros(sizU)
#     all_data_v = np.zeros(sizU)
#     all_data_w = np.zeros(sizU)
#     all_data_p = np.zeros(sizU)
    
#     all_data_e1 = np.zeros(sizU)
#     all_data_e2 = np.zeros(sizU)
#     all_data_e3 = np.zeros(sizU)
#     all_data_e4 = np.zeros(sizU)

#     # ==== 时间循环 ====
#     for count, tt in enumerate(tvec):
#         if count % 10 == 0:
#             print(f"--- predicting frame {count} / {len(tvec)} ---")

#         t_pred = tt * np.ones_like(x_pred)
#         pred_in = np.concatenate((t_pred, x_pred, y_pred, z_pred), axis=1)
#         pred_in = pred_in.astype(np.float32)   #


#         # 模型预测
#         u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred_in)
        
#         e1, e2, e3, e4 = pinn_model.ns_eqns(pred_in)

#         all_data_e1[:, :, count] = e1.numpy().reshape(sizU[0], sizU[1])
#         all_data_e2[:, :, count] = e2.numpy().reshape(sizU[0], sizU[1])
#         all_data_e3[:, :, count] = e3.numpy().reshape(sizU[0], sizU[1])
#         all_data_e4[:, :, count] = e4.numpy().reshape(sizU[0], sizU[1])

#         # e1 = e1.numpy()
#         # e2 = e2.numpy()
#         # e3 = e3.numpy()
#         # e4 = e4.numpy()
        
#         # 2D重构
#         all_data_u[:, :, count] = u_pred.reshape(sizU[0], sizU[1])
#         all_data_v[:, :, count] = v_pred.reshape(sizU[0], sizU[1])
#         all_data_w[:, :, count] = w_pred.reshape(sizU[0], sizU[1])
#         all_data_p[:, :, count] = p_pred.reshape(sizU[0], sizU[1])
        
#         # all_data_e1[:, :, count] = e1.reshape(sizU[0], sizU[1])
#         # all_data_e2[:, :, count] = e2.reshape(sizU[0], sizU[1])
#         # all_data_e3[:, :, count] = e3.reshape(sizU[0], sizU[1])
#         # all_data_e4[:, :, count] = e4.reshape(sizU[0], sizU[1])

#     # ==== 保存结果 ====
#     filepath = './predict_results'
#     os.makedirs(filepath, exist_ok=True)
#     filename = f"{hp['savename']}_plane_z0_predict.mat"
#     sio.savemat(os.path.join(filepath, filename),
#                 {'xmesh': pred_xmesh_2d,
#                  'ymesh': pred_ymesh_2d,
#                  'z0': z0,
#                  'all_data_u': all_data_u,
#                  'all_data_v': all_data_v,
#                  'all_data_w': all_data_w,
#                  'all_data_p': all_data_p,
#                  'all_data_e1': all_data_e1,
#                  'all_data_e2': all_data_e2,
#                  'all_data_e3': all_data_e3,
#                  'all_data_e4': all_data_e4})

#     print("Prediction on z=0 plane finished and saved.")
    
def predict_on_plane(pinn_model, coord1_mesh, coord2_mesh, const_coord, const_coord_name, tvec):
    """
    Predict flow field on a 2D plane
    Args:
        pinn_model: NS3D_UnSteady_PINNs model
        coord1_mesh: 2D array of first varying coordinate
        coord2_mesh: 2D array of second varying coordinate
        const_coord: scalar value of constant coordinate
        const_coord_name: 'x', 'y', or 'z' - which coordinate is constant
        tvec: time vector
    Returns:
        Dictionary with predicted u, v, w, p, e1, e2, e3, e4
    """
    # Flatten coordinates
    coord1_pred = coord1_mesh.flatten()[:, None]
    coord2_pred = coord2_mesh.flatten()[:, None]
    const_coord_pred = const_coord * np.ones_like(coord1_pred)

    # Initialize arrays
    sizU = coord1_mesh.shape + (len(tvec),)
    all_data_u = np.zeros(sizU)
    all_data_v = np.zeros(sizU)
    all_data_w = np.zeros(sizU)
    all_data_p = np.zeros(sizU)
    all_data_e1 = np.zeros(sizU)
    all_data_e2 = np.zeros(sizU)
    all_data_e3 = np.zeros(sizU)
    all_data_e4 = np.zeros(sizU)

    # Time loop prediction
    for count, tt in enumerate(tvec):
        if count % 10 == 0:
            print(f"--- predicting frame {count} / {len(tvec)} ---")

        t_pred = tt * np.ones_like(coord1_pred)

        # Build input based on which coordinate is constant
        if const_coord_name == 'x':  # x=constant, varying y and z
            pred_in = np.concatenate((t_pred, const_coord_pred, coord1_pred, coord2_pred), axis=1)
        elif const_coord_name == 'y':  # y=constant, varying x and z
            pred_in = np.concatenate((t_pred, coord1_pred, const_coord_pred, coord2_pred), axis=1)
        else:  # z=constant, varying x and y
            pred_in = np.concatenate((t_pred, coord1_pred, coord2_pred, const_coord_pred), axis=1)

        pred_in = pred_in.astype(np.float32)

        # Model prediction
        u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred_in)
        e1, e2, e3, e4= pinn_model.ns_eqns(pred_in)

        # Store results
        all_data_u[:, :, count] = u_pred.reshape(sizU[0], sizU[1])
        all_data_v[:, :, count] = v_pred.reshape(sizU[0], sizU[1])
        all_data_w[:, :, count] = w_pred.reshape(sizU[0], sizU[1])
        all_data_p[:, :, count] = p_pred.reshape(sizU[0], sizU[1])
        all_data_e1[:, :, count] = e1.numpy().reshape(sizU[0], sizU[1])
        all_data_e2[:, :, count] = e2.numpy().reshape(sizU[0], sizU[1])
        all_data_e3[:, :, count] = e3.numpy().reshape(sizU[0], sizU[1])
        all_data_e4[:, :, count] = e4.numpy().reshape(sizU[0], sizU[1])

    return {
        'all_data_u': all_data_u,
        'all_data_v': all_data_v,
        'all_data_w': all_data_w,
        'all_data_p': all_data_p,
        'all_data_e1': all_data_e1,
        'all_data_e2': all_data_e2,
        'all_data_e3': all_data_e3,
        'all_data_e4': all_data_e4
    }


def predict_3d3c_AOCFD():
    """
    2026-08-05 更新：适配"加噪声+降数据量"实验（合成噪声实验，回应审稿意见 Overall/C2）。
    网格与时间范围直接取自 noise5 数据文件（pred_xmesh/pred_ymesh/pred_zmesh/mint/maxt，
    与原始 3d3c_test_3dBendpipe1_Re1000Wo10.mat 一致），不再依赖旧的 SSS13 / ParaScan 数据。
    平面：z=0（对称面 x-y）、y=2.4（弯管中心截面 x-z）、x=1.7（弯管中心截面 y-z）。

    使用前：先跑完 train_3d3c_AOCFD 得到对应 savename 的权重；
    预测哪组就把 savename 中的 data{xxx} 改为对应值（与 train.py 一致）。
    """

    # ==== Load 3D mesh & time range from the noise5 data file itself ====
    mesh_pathname = './data/bendpipe'
    # ---- 修改这里切换数据量（与 train.py 对应）----
    mesh_filename = 'Re1000Wo10_noise5_data5_pinn.mat'
    # ------------------------------------------------
    print(f"Loading 3D mesh & time range from {mesh_filename}...")
    mesh_tmp = h5py.File(os.path.join(mesh_pathname, mesh_filename), 'r')
    full_xmesh = mesh_tmp['pred_xmesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    full_ymesh = mesh_tmp['pred_ymesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    full_zmesh = mesh_tmp['pred_zmesh'][:]  # Shape: (25, 101, 143) = (Nz, Nx, Ny)
    mint = float(np.transpose(mesh_tmp['mint']))
    maxt = float(np.transpose(mesh_tmp['maxt']))
    mesh_tmp.close()

    print(f"3D mesh shape: {full_xmesh.shape}")

    # ==== Verify mesh orientation ====
    # Shape (25, 101, 143) corresponds to (Nz, Nx, Ny)
    # Extract coordinate levels along correct dimensions
    z_levels = full_zmesh[:, 0, 0]  # z values along axis 0 (Nz dimension)
    x_levels = full_xmesh[0, :, 0]  # x values along axis 1 (Nx dimension)
    y_levels = full_ymesh[0, 0, :]  # y values along axis 2 (Ny dimension)

    print(f"Z range: [{np.min(z_levels)}, {np.max(z_levels)}], {len(np.unique(np.round(z_levels, 6)))} unique values")
    print(f"Y range: [{np.min(y_levels)}, {np.max(y_levels)}], {len(np.unique(np.round(y_levels, 6)))} unique values")
    print(f"X range: [{np.min(x_levels)}, {np.max(x_levels)}], {len(np.unique(np.round(x_levels, 6)))} unique values")
    print("\nDimension mapping: mesh shape (Nz, Nx, Ny) = (25, 101, 143)")
    print(f"Time range: [{mint}, {maxt}]")

    # ==== Extract planes ====
    # Target dimensions: z=0 (x-y: Nx×Ny), y=2.4 (x-z: Nx×Nz), x=1.7 (y-z: Ny×Nz)

    # 1. Extract z=0 plane (x-y mesh)
    # Extract along Nz dimension (axis 0)
    idx_z0 = np.argmin(np.abs(z_levels - 0.0))
    z0 = float(z_levels[idx_z0])
    print(f"\nExtracting z=0 plane at index {idx_z0}, z0={z0}")

    # Extract z=0 plane (x-y mesh) - shape (Nx, Ny) = (101, 143)
    # xmesh_z0_slice and ymesh_z0_slice are (101, 143)
    xmesh_z0_slice = full_xmesh[idx_z0, :, :]  # Shape: (101, 143)
    ymesh_z0_slice = full_ymesh[idx_z0, :, :]  # Shape: (101, 143)
    print(f"z=0 plane mesh shape: {xmesh_z0_slice.shape}")

    # 2. Extract y=2.4 plane (x-z mesh)
    # Extract along Ny dimension (axis 2)
    idx_y2p4 = np.argmin(np.abs(y_levels - 2.4))
    y_val = float(y_levels[idx_y2p4])
    print(f"\nExtracting y=2.4 plane at index {idx_y2p4}, y={y_val}")

    # Extract y=2.4 plane (x-z mesh) - shape (Nx, Nz) = (101, 25)
    # x varies along axis 1 (101), z varies along axis 0 (25)
    xmesh_y2p4 = full_xmesh[:, :, idx_y2p4]  # Shape: (25, 101) - (Nz, Nx)
    zmesh_y2p4 = full_zmesh[:, :, idx_y2p4]  # Shape: (25, 101) - (Nz, Nx)

    # Transpose to get (101, 25) = (Nx, Nz)
    xmesh_y2p4 = xmesh_y2p4.T  # (101, 25)
    zmesh_y2p4 = zmesh_y2p4.T  # (101, 25)

    print(f"y=2.4 plane mesh shape: {xmesh_y2p4.shape}")

    # 3. Extract x=1.7 plane (y-z mesh)
    # Extract along Nx dimension (axis 1)
    idx_x1p7 = np.argmin(np.abs(x_levels - 1.7))
    x_val = float(x_levels[idx_x1p7])
    print(f"\nExtracting x=1.7 plane at index {idx_x1p7}, x={x_val}")

    # Extract x=1.7 plane (y-z mesh) - shape should be (143, 25)
    # y varies along axis 2 (143), z varies along axis 0 (25)
    ymesh_x1p7 = full_ymesh[:, idx_x1p7, :]  # Shape: (25, 143) - (Nz, Ny)
    zmesh_x1p7 = full_zmesh[:, idx_x1p7, :]  # Shape: (25, 143) - (Nz, Ny)

    # Transpose to get (143, 25) = (Ny, Nz)
    ymesh_x1p7 = ymesh_x1p7.T  # (143, 25)
    zmesh_x1p7 = zmesh_x1p7.T  # (143, 25)

    print(f"x=1.7 plane mesh shape: {ymesh_x1p7.shape}")

    # ==== Load model ====
    print("\nLoading trained model...")
    savename = 'Re1000Wo10_13_156_noise5_data5_run0'
    save_file = './weights/' + savename + '/' + savename
    domain = sio.loadmat(save_file + '_paras.mat', squeeze_me=True)
    norm_paras = domain['norm_paras']

    hp = {
        'layers': [4] + 13 * [156] + [4],
        'ExistModel': 1,
        'train': False,
        'maptype': 'rnn',
        'savename': savename,
        'Re': 1000.0,
        'alpha': 1.0,
        'norm_paras': norm_paras,
        'tf_epochs': 8000,
        'tf_batch_size': 5000,
        'initial_epoch': 0,
        'init_lr': 1.0e-3,
        'bfgs_epochs': 0,
        'bfgs_batch_size': 10000,
        'lm_epochs': 0,
        'lm_batch_size': 200
    }

    pinn_model = NS3D_UnSteady_PINNs(hp)
    print("Model loaded successfully!")

    # ==== Setup time vector ====
    # 2026-08-05：mint/maxt 已在函数开头从 noise5 数据文件读取（mint=0, maxt=15.708，
    # 对应 Wo=10 的一个完整脉动周期），不再依赖旧的 ParaScan 数据。
    tvec = np.linspace(mint, maxt, 201)
    print(f"Time range: [{mint}, {maxt}], {len(tvec)} time steps")

    # ==== Predict on all three planes ====

    filepath = './predict_results'
    os.makedirs(filepath, exist_ok=True)

    # 1. Predict on z=0 plane (对称面 x-y)
    print("\n" + "="*50)
    print("Predicting on z=0 plane...")
    print("="*50)
    results_z0 = predict_on_plane(pinn_model, xmesh_z0_slice, ymesh_z0_slice, z0, 'z', tvec)

    # Save z=0 plane
    filename_z0 = f"{hp['savename']}_plane_z0_predict.mat"
    sio.savemat(os.path.join(filepath, filename_z0),
                {'xmesh': xmesh_z0_slice,
                 'ymesh': ymesh_z0_slice,
                 'z0': z0,
                 **results_z0})
    print(f"z=0 plane results saved to: {filename_z0}")

    # 2. Predict on y=2.4 plane
    print("\n" + "="*50)
    print("Predicting on y=2.4 plane...")
    print("="*50)
    results_y2p4 = predict_on_plane(pinn_model, xmesh_y2p4, zmesh_y2p4, y_val, 'y', tvec)

    # Save y=2.4 plane
    filename_y2p4 = f"{hp['savename']}_plane_y2p4_predict.mat"
    sio.savemat(os.path.join(filepath, filename_y2p4),
                {'xmesh': xmesh_y2p4,
                 'zmesh': zmesh_y2p4,
                 'y': y_val,
                 **results_y2p4})
    print(f"y=2.4 plane results saved to: {filename_y2p4}")

    # 3. Predict on x=1.7 plane
    print("\n" + "="*50)
    print("Predicting on x=1.7 plane...")
    print("="*50)
    results_x1p7 = predict_on_plane(pinn_model, ymesh_x1p7, zmesh_x1p7, x_val, 'x', tvec)

    # Save x=1.7 plane
    filename_x1p7 = f"{hp['savename']}_plane_x1p7_predict.mat"
    sio.savemat(os.path.join(filepath, filename_x1p7),
                {'ymesh': ymesh_x1p7,
                 'zmesh': zmesh_x1p7,
                 'x': x_val,
                 **results_x1p7})
    print(f"x=1.7 plane results saved to: {filename_x1p7}")

    print("\n" + "="*50)
    print("All predictions completed successfully!")
    print("="*50)
    print(f"Results saved in: {filepath}/")
    print(f"  - {filename_z0}")
    print(f"  - {filename_y2p4}")
    print(f"  - {filename_x1p7}")

                    
if __name__ == "__main__":
   
    # predict_3d3c_wanglei()
    
    # predict_2d2c_mitralvalve_fsi()
    
    # predict_3d3c_mitralvalve_fsi()
    
    # predict_2d2c_wufan_fsi()
    
    # predict_2d2c_wangyt_fsi()
    
    # predict_2d2c_liuyi_fsi()
    
    # predict_2d2c_zangzy()
    
    # predict_2d2c_wufan()

    # predict_2d2c_bendpipe()

    predict_3d3c_AOCFD()
