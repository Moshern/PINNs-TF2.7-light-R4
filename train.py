# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

from datagenerator import wufan_DataGenerator
from datagenerator import mitralvalve3D_DataGenerator
from pinns_2d import NS2D_UnSteady_PINNs
from pinns_3d import NS3D_UnSteady_PINNs
import matplotlib.pyplot as plt
import numpy as np


def train_2d2c_bendpipe_zxc():
    """
    修改自PINN for 2d2c of oscillatory cylinder
    用来处理bendpipe的计算结果
    """
    data_pathname = './data/bendpipe'
    rnum = [0]
    data_filename = '2dWo10Bpi32_reslu40_noise0_pinn.mat'
    # 相同的数据格式
    data, norm_paras = wufan_DataGenerator(data_pathname, data_filename)

    """
    parameters
    """
    for runidx in rnum:
        savename = '2dWo10Bpi32_13_128'
        hp = {'layers': [3] + 13 * [128] + [3],
              'ExistModel': 0,
              'train': True,
              'maptype': 'rnn',
              'savename': savename,
              'Re': 1000.0,
              'alpha': 1.0,
              'norm_paras': norm_paras,
              'tf_epochs': 6000,
              'tf_batch_size': 10000,
              'initial_epoch': 0,
              'init_lr': 1.0e-3,
              'bfgs_epochs': 0,
              'bfgs_batch_size': 10000,
              'lm_epochs': 0,
              'lm_batch_size': 200}

        pinn_model = NS2D_UnSteady_PINNs(hp)
        # 设置方程点
        pinn_model.set_eqns_points(data['data_eqns'])
        # 设置数据
        pinn_model.set_supervised_flow(data['data_supervised'])
        # 设置边界条件
        pinn_model.set_cond_bc('uv', data['bc_uv'])
        # pinn_model.set_cond_bc('uvw', data['bc_uvw'])
        # pinn_model.set_cond_bc('p', data['bc_p'])
        # pinn_model.set_cond_bc('py', data['bc_py'])
        # pinn_model.set_cond_bc('uy', data['bc_uy'])
        # pinn_model.set_cond_bc('vy', data['bc_vy'])
        # training and saving
        pinn_model.train()

    # plot
    plt.figure()
    # plt.axes(xscale="log", yscale="log")  # 同时设置横纵坐标均为对数坐标
    plt.axes(yscale="log")  # 同时设置横纵坐标均为对数坐标
    plt.ylim(1e-6, 1e2)
    N = np.arange(0,len(pinn_model.loss_all))
    # 生成从1开始的索引（避免0值）
    # N = np.arange(1, len(pinn_model.loss_all)+1)  # 修改这里：索引从1开始
    plt.plot(N,pinn_model.loss_all,label='total_loss')
    plt.plot(N,pinn_model.loss_fdat,label='fdat_loss')
    # plt.plot(N,pinn_model.loss_bdat,label='bdat_loss')
    plt.plot(N,pinn_model.loss_eqns,label='eqns_loss')
    plt.plot(N,pinn_model.loss_conds,label='conds_loss')
    plt.title('Training Loss')
    # plt.xlabel('Epoch #(log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(linestyle='-.')
    plt.legend(loc=1)

    save_file = './weights/'+pinn_model.savename+'/'+pinn_model.savename
    plt.savefig(save_file+'_loss.png', dpi=300)
    plt.show()


def train_3d3c_AOCFD():
    """
    PINN for 3d3c of AOCFD
    从2.7FSI版本的train_3d3c_mitralvalve_fsi中修改

    """
    data_pathname = './data/bendpipe'
    # ---- 修改这里切换数据量 ----
    data_filename = 'Re1000Wo10_noise5_data5_pinn.mat'
    # -----------------------------
    data, norm_paras = mitralvalve3D_DataGenerator(data_pathname, data_filename)

    # rnum = [0,1,2]
    rnum = [0]
    for runidx in rnum:
        """
        parameters
        """
        # savename 需与 data_filename 中的数据量对应
        savename = 'Re1000Wo10_13_156_noise5_data5_run{runidx}'.format(runidx=runidx)
        hp = {'layers': [4] + 13 * [156] + [4],
              'ExistModel': 0,
              'train': True,
              'maptype': 'rnn',
              'savename': savename,
              'Re': 1000.0,
              'alpha': 1.0,
              'norm_paras': norm_paras,
              'tf_epochs': 8000,
              'tf_batch_size': 5000,  # 之前是5000
              'initial_epoch': 0,
              'init_lr': 1.0e-3,
              'bfgs_epochs': 0,
              'bfgs_batch_size': 10000,
              'lm_epochs': 0,
              'lm_batch_size': 200}

        pinn_model = NS3D_UnSteady_PINNs(hp)
        # 设置方程点
        pinn_model.set_eqns_points(data['data_eqns'])
        # 设置边界条件
        pinn_model.set_cond_bc('uvw', data['bc_uvw'])
        # 设置数据
        pinn_model.set_data_supervised(data['data_supervised'])
        # 设置边界数据
        # pinn_model.set_supervised_neuralbc(data['bc_supervised'])
        # pinn_model.set_cond_bc('p', data['bc_p'])
        # pinn_model.set_cond_bc('py', data['bc_py'])
        # pinn_model.set_cond_bc('uy', data['bc_uy'])
        # pinn_model.set_cond_bc('vy', data['bc_vy'])
        # pinn_model.set_cond_bc('wy', data['bc_wy'])

        # training and saving
        pinn_model.train()

        # plot
    plt.figure()
    plt.axes(yscale="log")
    plt.ylim(1e-6, 1e2)
    N = np.arange(0, len(pinn_model.loss_all))
    plt.plot(N, pinn_model.loss_all, label='total_loss')
    plt.plot(N, pinn_model.loss_fdat, label='fdat_loss')
    # plt.plot(N,pinn_model.loss_bdat,label='bdat_loss')
    plt.plot(N, pinn_model.loss_eqns, label='eqns_loss')
    plt.plot(N, pinn_model.loss_conds, label='conds_loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(linestyle='-.')
    plt.legend(loc=1)

    save_file = './weights/' + pinn_model.savename
    plt.savefig(save_file + '_loss.png', dpi=300)
    plt.show()


if __name__ == "__main__":

    train_3d3c_AOCFD()



