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


def wufan_DataGenerator(data_pathname, data_filename):
    """
    Load the 2D bend-pipe training data.
    """
    # Please use HDF reader for matlab v7.3 files
    tmp = h5py.File(os.path.join(data_pathname, data_filename),'r')

    keys = ['data_eqns', 'data_supervised',
            'bc_uv', 'bc_init',
            'bc_u',  'bc_v', 'bc_p',
            'bc_ux', 'bc_uy',
            'bc_vx', 'bc_vy',
            'bc_px', 'bc_py',
            'minx','maxx','dx',
            'miny','maxy','dy']

    # default is no conditions
    data_dict = dict.fromkeys(keys, None)

    data_dict['data_eqns'] = np.transpose(tmp['data_eqns'])
    data_dict['data_supervised'] = np.transpose(tmp['data_supervised'])
    data_dict['bc_uv'] = np.transpose(tmp['bc_uv'])

    # normalization parameters
    # [min(t),min(x),min(y),mean(u),mean(v),mean(p)]
    # [max(t),max(x),max(y),std(u),std(v),std(p)]
    norm_paras = np.zeros([2,6])
    eqns = np.transpose(tmp['data_eqns'])
    norm_paras[0,0:3] = eqns[:,0:3].min(0)
    norm_paras[1,0:3] = eqns[:,0:3].max(0)
    data = data_dict['data_supervised']
    norm_paras[0,3:6] = np.mean(data[:,3:6], 0)
    norm_paras[1,3:6] = np.std(data[:,3:6], 0)
    norm_paras[0,3:4] = 0
    norm_paras[1,3:4] = 1
    norm_paras[0,4:5] = 0
    norm_paras[1,4:5] = 1
    norm_paras[0,5:6] = 0
    norm_paras[1,5:6] = 1

    return data_dict, norm_paras
