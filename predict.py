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


def predict_2d2c_bendpipe():
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

    count = -1
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
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = ('%s_predict.mat') % (hp['savename'])
    sio.savemat(os.path.join(filepath, filename), {'xmesh':pred_xmesh, 'ymesh':pred_ymesh,\
                                       'all_data_u':all_data_u,\
                                       'all_data_v':all_data_v,\
                                       'all_data_p':all_data_p,\
                                       'all_data_e1':all_data_e1,\
                                       'all_data_e2':all_data_e2,\
                                       'all_data_e3':all_data_e3})


if __name__ == "__main__":

    predict_2d2c_bendpipe()
