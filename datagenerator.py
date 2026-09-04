# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import random
import h5py
import os



random.seed(1234)
np.random.seed(1234)
    

def Hasanuzzaman_DataGenerator(data_pathname):
    """
    steady cylinder
    """
    # Please use HDF reader for matlab v7.3 files
    # data = sio.loadmat(os.path.join(data_pathname, data_filename))
    data_filename = '2d3c_hasanuzzaman.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    
    keys = ['data_eqns', 'data_supervised',
            'bc_uvw', 'bc_init',
            'bc_u',  'bc_v', 'bc_w', 'bc_p',
            'bc_ux', 'bc_uy', 'bc_uz',
            'bc_vx', 'bc_vy', 'bc_vz',
            'bc_wx', 'bc_wy', 'bc_wz',           
            'bc_px', 'bc_py', 'bc_pz',
            'minx','maxx','dx',
            'miny','maxy','dy']
    
    # default is no conditions
    data_dict = dict.fromkeys(keys, None)
    
    data_dict['data_eqns'] = np.transpose(tmp['data_eqns'])
    data_dict['data_supervised'] = np.transpose(tmp['data_supervised'])
    data_dict['bc_uvw'] = np.transpose(tmp['bc_uvw'])
        
    # normalization parameters
    # [min(t),min(x),min(y),mean(u),mean(v),mean(p)]
    # [max(t),max(x),max(y),std(u),std(v),std(p)]
    norm_paras = np.zeros([2,8])    
    eqns =  np.transpose(tmp['data_eqns'])
    norm_paras[0,0:4] = eqns[:,0:4].min(0)
    norm_paras[1,0:4] = eqns[:,0:4].max(0)
    data = data_dict['data_supervised']
    norm_paras[0,4:8] = np.mean(data[:,4:8], 0)
    norm_paras[1,4:8] = np.std(data[:,4:8], 0)
    norm_paras[0,7] = 0
    norm_paras[1,7] = 1
    
    return data_dict, norm_paras







def mitralvalve3D_DataGenerator(data_pathname, data_filename):
    """
    steady cylinder
    """
    # Please use HDF reader for matlab v7.3 files
    # data = sio.loadmat(os.path.join(data_pathname, data_filename))
    # data_filename = '2d2c_mitralvalve_healthy_vocation_refined400.mat'
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')
    
    keys = ['data_eqns', 'data_supervised','bc_supervised',
            'bc_uvw', 'bc_init','neuralbc',
            'bc_u',  'bc_v', 'bc_w', 'bc_p',
            'bc_ux', 'bc_uy', 'bc_uz',
            'bc_vx', 'bc_vy', 'bc_vz',
            'bc_wx', 'bc_wy', 'bc_wz',           
            'bc_px', 'bc_py', 'bc_pz',
            'minx','maxx','dx',
            'miny','maxy','dy']
    
    # default is no conditions
    data_dict = dict.fromkeys(keys, None)
    
    # data_dict['data_eqns'] = np.transpose(tmp['data_eqns'])
    # data_dict['data_supervised'] = np.transpose(tmp['data_supervised'])
    # data_dict['bc_supervised'] = np.transpose(tmp['bc_supervised'])
    # data_dict['neuralbc'] = np.transpose(tmp['neuralbc'])
        
    data_dict['data_eqns'] = np.transpose(tmp['data_eqns'])
    data_dict['data_supervised'] = np.transpose(tmp['data_supervised'])
    data_dict['bc_uvw'] = np.transpose(tmp['bc_uvw'])
    # data_dict['bc_p'] = np.transpose(tmp['bc_p'])
    # data_dict['bc_uy'] = np.transpose(tmp['bc_uy'])
    # data_dict['bc_vy'] = np.transpose(tmp['bc_vy'])
    # data_dict['bc_wy'] = np.transpose(tmp['bc_wy'])
    # data_dict['bc_py'] = np.transpose(tmp['bc_py'])

    # normalization parameters
    # [min(t),min(x),min(y),mean(u),mean(v),mean(p)]
    # [max(t),max(x),max(y),std(u),std(v),std(p)]
    norm_paras = np.zeros([2,8])    
    eqns =  np.transpose(tmp['data_eqns'])
    norm_paras[0,0:4] = eqns[:,0:4].min(0)
    norm_paras[1,0:4] = eqns[:,0:4].max(0)
    data = data_dict['data_supervised']
    norm_paras[0,4:8] = np.mean(data[:,4:8], 0)
    norm_paras[1,4:8] = np.std(data[:,4:8], 0)

    norm_paras[0, 4] = 0
    norm_paras[1, 4] = 1
    norm_paras[0, 5] = 0
    norm_paras[1, 5] = 1
    norm_paras[0, 6] = 0
    norm_paras[1, 6] = 1

    norm_paras[0, 7] = 0
    norm_paras[1, 7] = 1
    

    return data_dict, norm_paras


"""
for testing
"""
if __name__ == "__main__":
    data_pathname = './data/hasanuzzaman'
    data, norm_paras = Hasanuzzaman_DataGenerator(data_pathname)  
    
