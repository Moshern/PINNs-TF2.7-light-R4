    # -*- coding: utf-8 -*-
"""
Created Jun14 2021
Custom training

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
from userbackend import tf, _GPU_NUM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
import tensorflow.keras.backend as K
from maps import generator
import scipy.io as sio
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize 
from funcs import LevenbergMarquardt, generate_dataset
import os
import time

"""
user defined informantion print
"""
class LossPrintingCallback(Callback):
    def __init__(self):
        super(LossPrintingCallback, self).__init__()
    """
    user defined loss print function
    """
    def on_epoch_begin(self, epoch, logs=None):
        """
        do not output anything
        """
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print("Epoch %05d: loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e, learning rate: %8.6f" %
                (epoch, logs["loss"], logs["loss_fn_data"], logs["loss_fn_eqns"], logs["loss_fn_conds"], lr)
        )
        
        
"""
user defined learning rate for optimizer
"""       
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, end_learning_rate, total_epochs, last_epochs, decay_epochs, steps_per_epoch, decay_rate):
      super().__init__()
      self.initial_learning_rate = initial_learning_rate
      self.end_learning_rate = end_learning_rate
      self.total_epochs = total_epochs
      self.last_epochs = last_epochs
      self.decay_epochs = decay_epochs
      self.steps_per_epoch = steps_per_epoch
      self.total_steps = total_epochs*steps_per_epoch
      self.decay_steps = decay_epochs*steps_per_epoch
      self.decay_rate = decay_rate
      self.grow_rate = 1.0/self.decay_rate
    
  def __call__(self, step):
      # the learning rate changes with steps not epoch
      #step_n = np.ceil(np.log(1.0-epoch/self.decay_epochs*(1.0-self.grow_rate))/np.log(self.grow_rate))-1.0
      curstep = step+self.last_epochs*self.steps_per_epoch
      step_n = tf.math.floor(curstep/self.decay_steps)
      lr_e = self.initial_learning_rate * tf.math.pow(self.decay_rate, step_n)
      lr_e = tf.cond(lr_e < self.end_learning_rate, lambda:self.end_learning_rate, lambda:lr_e)
      # the lr is set to 1e-5 for the last 1000 epochs
      lr = tf.cond(curstep > self.total_steps-1000*self.steps_per_epoch, lambda:self.end_learning_rate, lambda:lr_e)
      
      return lr 

 
    
 
    
class NS3D_UnSteady_PINNs(object):
    def __init__(self, paras):
        """
        hp: parameters            
        Note: equation points should include the points of boundary conditions
        """
        # clear session
        K.clear_session()
        # hp is the structure of hyper-parameters
        self.dtype = 'float32'
        self.layers = paras['layers']
        self.ExistModel = paras['ExistModel']
        self.maptype = paras['maptype']
        self.savename = paras['savename']
        self.Re = paras['Re'] 
        self.training = paras['train']
        self.tf_epochs = paras['tf_epochs']
        self.tf_batch_size = paras['tf_batch_size']
        self.initial_epoch = paras['initial_epoch']
        self.init_lr = paras['init_lr']
        self.bfgs_epochs = paras['bfgs_epochs']
        self.bfgs_batch_size = paras['bfgs_batch_size']
        self.lm_epochs = paras['lm_epochs']
        self.lm_batch_size = paras['lm_batch_size']
        self.data_rate = 1.0
        # record the iteration number
        self.iternum = 0
        self.steps_per_epoch = 30
        
        # use float64 by default
        K.set_floatx(self.dtype)

        # initialize boundary conditions
        self.conds = self.init_conditions()
        # initialize the data
        self.data_eqns = None
        self.data_sp = None
        
        # Initialize the loss recording list
        self.loss_all = []
        self.loss_fdat = []
        self.loss_eqns = []
        self.loss_conds = []
        # 根据梯度计算权重，初始值为0
        self.weight_fdat = [np.array(1.0)]
        self.weight_bc = [np.array(1.0)]
        

        # Multi GPU or single GPU
        if _GPU_NUM > 1:
            # tf.distribute.NcclAllReduce() only for linux
            self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        elif _GPU_NUM == 1:
            self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            
        self.gpus_number = self.strategy.num_replicas_in_sync 
        self.global_batch_size = self.tf_batch_size*self.gpus_number  
        # the batch size for computing the conditions
        self.cond_batch_size = int(self.global_batch_size/2.0)
        
        # whether to continue to training
        if self.ExistModel == 0:
            # without continue to training
            # epoch begins from 0
            self.epochs_last = 0
            
        elif self.ExistModel == 1:
            # with continue to training
            # read the last parameters
            self.loadParas(index='final') 
            self.tf_epochs = self.tf_epochs+self.epochs_last            
    
        # learning rate
        reduce_lr = MyLRSchedule(self.init_lr, 1e-5, self.tf_epochs, self.epochs_last, 50, self.steps_per_epoch, 0.95)

        # generate the model and optimizer within strategy
        if self.training:
            with self.strategy.scope():
                # optimizer
                self.optimizer = Adam(reduce_lr)  
                if self.ExistModel == 0:
                    self.norm_paras = paras['norm_paras']
                    self.model = self.build_model()
                    
                elif self.ExistModel == 1:
                    self.model = self.loadNN()
        else:
            # optimizer
            self.optimizer = Adam(reduce_lr)  
            if self.ExistModel == 0:
                self.norm_paras = paras['norm_paras']
                self.model = self.build_model()
                
            elif self.ExistModel == 1:
                self.model = self.loadNN()            
                    
            
        self.model.summary()
                
        

    
    
    def build_model(self):
        model = generator(self.layers, self.norm_paras, maptype=self.maptype)
     
        return model
    
 
    
    def init_conditions(self):
        """
        initilize the condition dictionary
        """
        conds_keys = ['init', 'uvw',
                      'u',  'v', 'w', 'p',
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        
        # default is no conditions
        conds_dict = dict.fromkeys(conds_keys, None)
       
        return conds_dict
            
            
            
            
    def set_cond_bc(self, keys, X=None):
        """
        set the Dirichlet BCs
        initial data: X=[t,x,y,z,u,v,w,p]
        Dirichlet BC of uvw: X=[t,x,y,z,u,v,w]
        Dirichlet BC of X: X = [t,x,y,z,x]
        conds_keys = ['init', 'uvw',
                      'u',  'v', 'w', 'p',
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        keys: 'init', 'uvw', 'u',  'v', 'w', 'p'
        """    
        """
        set the Neumann BCs
        Neumann BC of X: X=[t,x,y,z,x]
        conds_keys = ['init', 'uvw',
                      'u',  'v', 'w', 'p',
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        keys: 'ux', 'uy', 'uz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'px', 'py', 'pz']
        """         
        key = keys.lower()
        if X is None:
            self.conds[key] = None
        else:
            train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                    data_rate=1.0,
                                                    batchsize=self.cond_batch_size,
                                                    isrepeat=True)
            # 分布式训练
            train_data = self.strategy.experimental_distribute_dataset(train_data)
            val_data = self.strategy.experimental_distribute_dataset(val_data)
            # 使用iter显示创建迭代器            
            self.conds[key]  = [train_data, iter(train_data)]           
  
     


    def set_eqns_points(self, X=None):
        """
        set the data points for equations
        data equations: X=[t,x,y,z]
        """ 
        # generate the dataset
        if X is None:
            self.data_eqns = None
        else:
            train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                    data_rate=self.data_rate,
                                                    batchsize=self.global_batch_size,
                                                    isrepeat=True)
            # 分布式训练
            train_data = self.strategy.experimental_distribute_dataset(train_data)
            val_data = self.strategy.experimental_distribute_dataset(val_data) 
            # estimate the batch number
            # self.batch_num = int(X.shape[0]*self.data_rate/self.global_batch_size)
            # 使用iter显示创建迭代器
            self.data_eqns = [train_data, iter(train_data)]

 


    def set_data_supervised(self, X=None):
        """
        set the data points for supervising
        data supervised: X=[t,x,y,u,v,p]
        """   
        if X is None:
            self.data_sp = None
        else:
             train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                     data_rate=1.0,
                                                     batchsize=self.cond_batch_size,
                                                     isrepeat=True)
             # 分布式训练
             train_data = self.strategy.experimental_distribute_dataset(train_data)
             val_data = self.strategy.experimental_distribute_dataset(val_data)
             # 使用iter显示创建迭代器
             self.data_sp = [train_data, iter(train_data)]
             

           
                
    @tf.function
    def ns_eqns(self, X):
        """
        Returns
        -------
        residual of Navier-Stokes equations

        """    
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint             
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            tape2.watch(x)
            tape2.watch(y)
            tape2.watch(z)
                
            with tf.GradientTape(persistent=True) as tape1:
                # Watching gradients of t,x,y,z
                tape1.watch(t)
                tape1.watch(x)
                tape1.watch(y)
                tape1.watch(z)
                # Packing together the inputs
                X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
                # Getting the prediction
                # Y = self.model(X, training=self.training)
                # Y: [u,v,p,um,vm,pm,ustd,vstd,pstd]
                Y = self.model(X)
                u = Y[:,0:1]
                v = Y[:,1:2]
                w = Y[:,2:3]
                p = Y[:,3:4]
                # recover to real value
                u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
                v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
                w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
                p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
            # first-order deriavative
            u_t = tape1.gradient(u, t)
            v_t = tape1.gradient(v, t)
            w_t = tape1.gradient(w, t)
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_z = tape1.gradient(u, z)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            v_z = tape1.gradient(v, z)
            w_x = tape1.gradient(w, x)
            w_y = tape1.gradient(w, y)
            w_z = tape1.gradient(w, z)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            p_z = tape1.gradient(p, z)
        # second-order deriavative
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        u_zz = tape2.gradient(u_z, z)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        v_zz = tape2.gradient(v_z, z)
        w_xx = tape2.gradient(w_x, x)
        w_yy = tape2.gradient(w_y, y)
        w_zz = tape2.gradient(w_z, z)      
        
        e1 = u_t + (u * u_x + v * u_y + w * u_z) + p_x - (1.0 / self.Re) * (u_xx + u_yy + u_zz)
        e2 = v_t + (u * v_x + v * v_y + w * v_z) + p_y - (1.0 / self.Re) * (v_xx + v_yy + v_zz)
        e3 = w_t + (u * w_x + v * w_y + w * w_z) + p_z - (1.0 / self.Re) * (w_xx + w_yy + w_zz)
        e4 = u_x + v_y + w_z

        # Letting the tape go
        del tape1, tape2
        # Buidling the PINNs
        return e1, e2, e3, e4
    

  
    @tf.function
    def get_gradient(self, X, flag):
        """
        Returns
        -------
        the gradient of u,v,p

        """
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        z = tf.convert_to_tensor(X[:,3:4], self.dtype)

        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint
        with tf.GradientTape(persistent=True) as tape:
            # Watching gradients of t,x,y,z
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            # Packing together the inputs
            X = tf.stack([t[:,0],x[:,0],y[:,0],z[:,0]], axis=1)
            # Getting the prediction
            # Y = self.model(X, training=self.training)
            Y = self.model(X)
            u = Y[:,0:1]
            v = Y[:,1:2]
            w = Y[:,2:3]
            p = Y[:,3:4]
            # recover to real value
            u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
            v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
            w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
            p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
        # first-order deriavative
        if flag.lower() == 'ux':
            g = tape.gradient(u, x)
        elif flag.lower() == 'uy':
            g = tape.gradient(u, y)
        elif flag.lower() == 'uz':
            g = tape.gradient(u, z)
        elif flag.lower() == 'vx':
            g = tape.gradient(v, x)
        elif flag.lower() == 'vy':
            g = tape.gradient(v, y)
        elif flag.lower() == 'vz':
            g = tape.gradient(v, z)
        elif flag.lower() == 'wx':
            g = tape.gradient(w, x)
        elif flag.lower() == 'wy':
            g = tape.gradient(w, y)
        elif flag.lower() == 'wz':
            g = tape.gradient(w, z)
        elif flag.lower() == 'px':
            g = tape.gradient(p, x)
        elif flag.lower() == 'py':
            g = tape.gradient(p, y)
        elif flag.lower() == 'pz':
            g = tape.gradient(p, z)
        # else:
        #     valid_flags = ['ux', 'uy', 'uz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'px', 'py', 'pz']

            # Letting the tape go
        del tape

        # Buidling the PINNs
        return g

    # def get_gradient(self, X, flag):
    #     """
    #     Returns
    #     -------
    #     the gradient of u, v, w, p (三维版本)
    #     """
    #     # 定义所有可能的梯度键（包含三维分量）
    #     keys = ['ux', 'uy', 'uz',
    #             'vx', 'vy', 'vz',
    #             'wx', 'wy', 'wz',
    #             'px', 'py', 'pz']
    #     grad = dict.fromkeys(keys, None)  # 初始化字典
    #
    #     t = tf.convert_to_tensor(X[:, 0:1], self.dtype)
    #     x = tf.convert_to_tensor(X[:, 1:2], self.dtype)
    #     y = tf.convert_to_tensor(X[:, 2:3], self.dtype)
    #     z = tf.convert_to_tensor(X[:, 3:4], self.dtype)
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(t)
    #         tape.watch(x)
    #         tape.watch(y)
    #         tape.watch(z)
    #         # 四维输入 [t, x, y, z]
    #         X = tf.stack([t[:, 0], x[:, 0], y[:, 0], z[:, 0]], axis=1)
    #         Y = self.model(X)
    #         u = Y[:, 0:1]
    #         v = Y[:, 1:2]
    #         w = Y[:, 2:3]
    #         p = Y[:, 3:4]
    #         # 反归一化
    #         u = u * self.norm_paras[1, 4] + self.norm_paras[0, 4]
    #         v = v * self.norm_paras[1, 5] + self.norm_paras[0, 5]
    #         w = w * self.norm_paras[1, 6] + self.norm_paras[0, 6]
    #         p = p * self.norm_paras[1, 7] + self.norm_paras[0, 7]
    #
    #     flag = flag.lower()
    #     # 处理单个梯度请求
    #     if flag == 'ux':
    #         grad['ux'] = tape.gradient(u, x)
    #     elif flag == 'uy':
    #         grad['uy'] = tape.gradient(u, y)
    #     elif flag == 'uz':
    #         grad['uz'] = tape.gradient(u, z)
    #     elif flag == 'vx':
    #         grad['vx'] = tape.gradient(v, x)
    #     elif flag == 'vy':
    #         grad['vy'] = tape.gradient(v, y)
    #     elif flag == 'vz':
    #         grad['vz'] = tape.gradient(v, z)
    #     elif flag == 'wx':
    #         grad['wx'] = tape.gradient(w, x)
    #     elif flag == 'wy':
    #         grad['wy'] = tape.gradient(w, y)
    #     elif flag == 'wz':
    #         grad['wz'] = tape.gradient(w, z)
    #     elif flag == 'px':
    #         grad['px'] = tape.gradient(p, x)
    #     elif flag == 'py':
    #         grad['py'] = tape.gradient(p, y)
    #     elif flag == 'pz':
    #         grad['pz'] = tape.gradient(p, z)
    #     # 处理 "all" 请求（返回所有梯度）
    #     elif flag == 'all':
    #         grad['ux'] = tape.gradient(u, x)
    #         grad['uy'] = tape.gradient(u, y)
    #         grad['uz'] = tape.gradient(u, z)
    #         grad['vx'] = tape.gradient(v, x)
    #         grad['vy'] = tape.gradient(v, y)
    #         grad['vz'] = tape.gradient(v, z)
    #         grad['wx'] = tape.gradient(w, x)
    #         grad['wy'] = tape.gradient(w, y)
    #         grad['wz'] = tape.gradient(w, z)
    #         grad['px'] = tape.gradient(p, x)
    #         grad['py'] = tape.gradient(p, y)
    #         grad['pz'] = tape.gradient(p, z)
    #     else:
    #         # 无效 flag 抛出异常
    #         valid_flags = keys + ['all']
    #         raise ValueError(f"Invalid flag: {flag}. Supported flags are: {valid_flags}")
    #
    #     del tape
    #     return grad  # 返回梯度字典

    
    
    @tf.function
    def get_uvwp(self, X):
        """
        get the output of the network
        The predict function is designed for performance in large scale inputs. 
        For small amount of inputs that fit in one batch, directly using
        __call__() is recommended for faster execution, e.g., model(x), 
        or model(x, training=False)

        """ 
        Xi = tf.convert_to_tensor(X[:,0:4], self.dtype)
        Y = self.model(Xi)
        u = Y[:,0:1]*self.norm_paras[1,4]+self.norm_paras[0,4]
        v = Y[:,1:2]*self.norm_paras[1,5]+self.norm_paras[0,5]
        w = Y[:,2:3]*self.norm_paras[1,6]+self.norm_paras[0,6]
        p = Y[:,3:4]*self.norm_paras[1,7]+self.norm_paras[0,7]
        
        return u, v, w, p   
 
    
 
    
    def get_dynamic_weight(self, batch_data):
        """
        动态计算权重
        为了节约内存，分别计算梯度
        """
        with tf.GradientTape(persistent=False) as tap:
            # loss of equation
            le = self.loss_fn_eqns(batch_data)
        # 方程误差关于神经网络参数的梯度
        ge = tap.gradient(le,
                          self.model.trainable_variables,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
        ge = [tf.reshape(j, (1, -1)) for j in ge]
        ge = tf.concat(ge, axis=1)
        ge_mean = tf.reduce_mean(tf.abs(ge))/tf.sqrt(le)
        # for tmp in ge:
        #     ge_list(tf.reduce_mean(tf.abs(tmp)))
        del tap
        
        with tf.GradientTape(persistent=False) as tap:
            # loss of data
            ld = self.loss_fn_data(batch_data)    
        # 数据误差关于神经网络参数的梯度
        gd = tap.gradient(ld,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gd = [tf.reshape(j, (1, -1)) for j in gd]
        gd = tf.concat(gd, axis=1)
        gd_mean = tf.reduce_mean(tf.abs(gd))/tf.sqrt(ld)
        # for tmp in gd:
        #     gd_list(tf.reduce_mean(tf.abs(tmp)))
        del tap
        
        with tf.GradientTape(persistent=False) as tap:
            # loss of boundary condtions
            lb = self.loss_fn_conds(batch_data)
        # 边界条件关于神经网络参数的梯度
        gb = tap.gradient(lb,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)

        gb = [tf.reshape(j, (1, -1)) for j in gb]
        gb = tf.concat(gb, axis=1)
        gb_mean = tf.reduce_mean(tf.abs(gb))/tf.sqrt(lb)
        del tap
        
        # 根据梯度计算权重, 更偏向数据   
        wd = tf.minimum(ge_mean/gd_mean,1.0e3)
        wb = tf.minimum(ge_mean/gb_mean,1.0e3)           
        
        return wd, wb
    
    
    
    @tf.function
    def grad_custom(self, batch_data, weight_data, weight_bc):
        """
        根据输入X=[t,x,y,z]计算graient
        """ 
        self.iternum = self.iternum+1          
        # estimate the total loss
        with tf.GradientTape(persistent=False) as tap:

            le = self.loss_fn_eqns(batch_data)
            # loss of data
            ld = self.loss_fn_data(batch_data)    
            # loss of boundary condtions
            lb = self.loss_fn_conds(batch_data)
            # 注意：动态权重在训练时提前调用
            ls = le + weight_data*ld + weight_bc*lb
            # ls = le + weight_bc*lb
 
        g = tap.gradient(ls,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        del tap
        return g, ls, le, ld, lb
    
        

        
    def loss_fn_all(self, Y_true, Y_pred):
        """
        自定义loss
        Y_true: [t, x, y]
        """       
        # loss of equations
        le = self.loss_fn_eqns(Y_true, Y_pred)
        # loss of data
        ld = self.loss_fn_data(Y_true, Y_pred)    
        # loss of boundary condtions
        lb = self.loss_fn_conds(Y_true, Y_pred)
        # total loss                          
        return le+ld+lb
    
    
    
    
    @tf.function
    def loss_custom(self, X, Y):
        # prediction
        Y_pred = self.model(X[:,0:4])
       # Y_pred = self.model(X[:,0:3])
        #注意：Y_true是[t,x,y]坐标
        Y_true = Y[:,0:6]
        
        le = self.loss_fn_eqns(Y_true, Y_pred)
        # loss of data
        ld = self.loss_fn_data(Y_true, Y_pred)    
        # loss of boundary condtions
        lb = self.loss_fn_conds(Y_true, Y_pred)
        # total loss
        ls = le+ld+lb
        return [ls, le, ld, lb]
 
    
 
    def loss_fn_eqns(self, data):
        """
        return the loss of equations
        """   
        Y_true = data['eqns'][0]
        Y_true = Y_true[:,0:6]
        # 首先计算方程点的loss
        # Y_true is the same as Xs
        X = Y_true[:,0:4]
        # the weight for space
        ws = Y_true[:,4:5]
        # the weight for time
        wt = Y_true[:,5:6]
        e1,e2,e3,e4 = self.ns_eqns(X)          
        e1 = e1 * ws * wt
        e2 = e2 * ws * wt
        e3 = e3 * ws * wt
        e4 = e4 * ws * wt
        # multi-gpu：总和除以global_batch_size
        loss_eqns_e1 = tf.reduce_sum(tf.square(e1))*(1.0/self.global_batch_size)
        loss_eqns_e2 = tf.reduce_sum(tf.square(e2))*(1.0/self.global_batch_size)
        loss_eqns_e3 = tf.reduce_sum(tf.square(e3))*(1.0/self.global_batch_size)
        loss_eqns_e4 = tf.reduce_sum(tf.square(e4))*(1.0/self.global_batch_size)
        loss_eqns = loss_eqns_e1+loss_eqns_e2+loss_eqns_e3+loss_eqns_e4

        return loss_eqns
    
    
    
    def loss_fn_conds(self, data):
        """
        return the loss of all the BCs
        # ii=0, initial BC,             [t,x,y,u,v,p]
        # ii=1, Dirichlet BC,           [t,x,y,u,v,p]
        # ii=2, Neumann BC of ux,       [t,x,y,ux]
        # ii=3, Neumann BC of uy,       [t,x,y,uy]
        # ii=4, Neumann BC of vx,       [t,x,y,vx]
        # ii=5, Neumann BC of vy,       [t,x,y,vy]
        # ii=6, Neumann BC of px,       [t,x,y,px]
        # ii=7, Neumann BC of py,       [t,x,y,py]   
        """             
        loss = 0.0
        # iterate to estimate the loss of Bcs
        for key, val in self.conds.items():
            if (key == 'init') or (key == 'uvw'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    ut = batch_x[:,4:5]
                    vt = batch_x[:,5:6]
                    wt = batch_x[:,6:7]
                    # pt = val[idx,5:6]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    
                    # 根据误差大小设置权重
                    tmpu = tf.square(ut-up)
                    tmpv = tf.square(vt-vp)
                    tmpw = tf.square(wt-wp)
                    # multi-gpu：总和除以global_batch_size
                    lossu = tf.reduce_sum(tmpu)*(1.0/self.cond_batch_size)
                    lossv = tf.reduce_sum(tmpv)*(1.0/self.cond_batch_size)   
                    lossw = tf.reduce_sum(tmpw)*(1.0/self.cond_batch_size)  
                    
                    loss = loss + 1.0*(lossu + lossv + lossw)
                                
                    
            elif (key == 'u'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    ut = batch_x[:,4:5]
                    # pt = val[idx,5:6]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    # 根据误差大小设置权重
                    tmp = tf.square(ut-up)
                    # multi-gpu：总和除以global_batch_size
                    lossu = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size) 
                    loss = loss + lossu
                    
            elif (key == 'v'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    vt = batch_x[:,4:5]
                    # pt = val[idx,5:6]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    # 根据误差大小设置权重
                    tmp = tf.square(vt-vp)    
                    # multi-gpu：总和除以global_batch_size
                    lossv = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size) 
                    loss = loss + lossv   
 
            elif (key == 'w'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    wt = batch_x[:,4:5]
                    # pt = val[idx,5:6]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    # 根据误差大小设置权重
                    tmp = tf.square(wt-wp)   
                    # multi-gpu：总和除以global_batch_size
                    lossw = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size) 
                    loss = loss + lossw                    

            elif (key == 'p'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    pt = batch_x[:,4:5]
                    # pt = val[idx,5:6]
                    up, vp, wp, pp = self.get_uvwp(tmpX)
                    # 根据误差大小设置权重
                    tmp = tf.square(pt-pp)
                    # multi-gpu：总和除以global_batch_size                   
                    lossp = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size)
                    loss = loss + lossp   
                                
            # for the conditions of gradients: ux, uy, uz, vx, vy, vz, px, py, pz
            else:
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:4]
                    gt = batch_x[:,4:5]
                    gp = self.get_gradient(tmpX, key)
                    # 根据误差大小设置权重
                    tmp = tf.square(gt-gp)
                    # multi-gpu：总和除以global_batch_size  
                    tmp = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size)
                    loss = loss + tmp

                    # # Debug output: Print key and loss contribution
                    # tf.print("Debug: key =", key, ", loss contribution =", tmp)
                    
        return loss
    


    def loss_fn_data(self, data):
        """
        return the loss of supervised data
        """     
        if self.data_sp is None:
            loss = 0.0
        else:
            x = data['data'][0]
            y = data['data'][1]
            # using the fluctuation to calculate the error
            tmpX = x[:,0:8]           
            # prediction
            up, vp, wp, pp = self.get_uvwp(tmpX[:,0:4])
            
            # get the data value
            ut = tmpX[:,4:5]
            vt = tmpX[:,5:6]
            wt = tmpX[:,6:7]
            # pt = tmpX[:,7:8]
            # multi-gpu：总和除以global_batch_size     
            lossu = tf.reduce_sum(tf.square(ut-up))*(1.0/self.cond_batch_size)
            lossv = tf.reduce_sum(tf.square(vt-vp))*(1.0/self.cond_batch_size)
            lossw = tf.reduce_sum(tf.square(wt-wp))*(1.0/self.cond_batch_size)
            # lossp = tf.reduce_sum(tf.square(pt-pp))*(1.0/self.cond_batch_size)

            loss = lossu+lossv+lossw
        return loss 
    
  
  


    def get_batch_data(self):
        """
        record the batch data for distribute training
        """
        batch_keys = ['eqns',
                      'data','init', 'uvw',
                      'u',  'v', 'w', 'p',
                      'ux', 'uy', 'uz',
                      'vx', 'vy', 'vz',
                      'wx', 'wy', 'wz',
                      'px', 'py', 'pz']
        
        # default is no conditions
        batch_dict = dict.fromkeys(batch_keys, None)
        # record
        if self.data_eqns is not None:
            it = self.data_eqns[1]
            batch_x, batch_y = next(it)
            batch_dict['eqns'] = [batch_x, batch_y]
            
        if self.data_sp is not None:
            it = self.data_sp[1]
            batch_x, batch_y = next(it)
            batch_dict['data'] = [batch_x, batch_y]
            
        # iterate to get batch BCs
        for key, val in self.conds.items():
            if val is not None:
                it = val[1]
                batch_x, batch_y = next(it)
                batch_dict[key] = [batch_x, batch_y]
        
        return batch_dict
        
        
  
    def train(self):
        if self.tf_epochs > 0:
            self.train_tf()
            
        if self.bfgs_epochs > 0:
            self.train_bfgs()    
 

       
    def train_tf(self):
        """
        trained by tensorflow
        """
        ###############################################################################################
        @tf.function
        def distributed_train_step(batch_data, wd, wb): 
            # run with the distribute input
            per_batch_ls, per_batch_le, per_batch_ld, per_batch_lb = \
                self.strategy.run(train_step, args=(batch_data, wd, wb))
            # fetch the loss
            batch_ls = self.strategy.reduce(tf.distribute.ReduceOp.SUM, \
                                            per_batch_ls,axis=None)
            batch_le = self.strategy.reduce(tf.distribute.ReduceOp.SUM, \
                                            per_batch_le,axis=None)
            batch_ld = self.strategy.reduce(tf.distribute.ReduceOp.SUM, \
                                            per_batch_ld,axis=None)
            batch_lb = self.strategy.reduce(tf.distribute.ReduceOp.SUM, \
                                            per_batch_lb,axis=None)                           
            return batch_ls, batch_le, batch_ld, batch_lb
        #################################################################################################
        
        ###############################################################################################
        def train_step(batch_data, wd, wb):
            # wd is the weight of data
            # wb is the weight of BC
            # estimate the loss and gradient
            batch_grad, batch_ls, batch_le, batch_ld, batch_lb = self.grad_custom(batch_data, wd, wb)
            # gradient optimize
            self.optimizer.apply_gradients(zip(batch_grad, self.model.trainable_variables))
            return batch_ls, batch_le, batch_ld, batch_lb
        #################################################################################################
                     
        self.training = True         
        epochs = self.tf_epochs   
        # main loop
        for epoch in np.arange(self.epochs_last, epochs):
            # record the time
            t1 = time.time()           
            # record the total loss
            epoch_ls_ave = tf.keras.metrics.Mean()
            # record the equation loss           
            epoch_le_ave = tf.keras.metrics.Mean()
            # record the data loss                
            epoch_ld_ave = tf.keras.metrics.Mean()
            # record the boundary condition loss                
            epoch_lb_ave = tf.keras.metrics.Mean()
            
            #self.optimizer.iterations(0)
            for count in np.arange(0, self.steps_per_epoch, 1):
                # get the batch data and BCs
                batch_data = self.get_batch_data()       
                if count == 0:
                    # estimate the dynamic weights which have been recorded in self.weight_data and self.weight_bc              
                    wd,wb = self.strategy.run(self.get_dynamic_weight, args=(batch_data,))
                    # 多个GPU需要进行平均
                    wd = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, \
                                              wd,axis=None)
                    wb = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, \
                                              wb,axis=None)
                    
                    if epoch == 0:
                        # use the initial values
                        wd = (1-0.0)*self.weight_fdat[-1]+0.0*K.get_value(wd)
                        wd = wd.astype(self.dtype)
                        wb = (1-0.0)*self.weight_bc[-1]+0.0*K.get_value(wb)
                        wb = wb.astype(self.dtype)
                        
                    elif epoch > 0:               
                        # record the weight: alpha = (1-lambda)*alpha+lambda*alpha
                        wd = (1-0.1)*self.weight_fdat[-1]+0.1*K.get_value(wd)
                        wd = wd.astype(self.dtype)
                        self.weight_fdat.append(wd)
                        wb = (1-0.1)*self.weight_bc[-1]+0.1*K.get_value(wb)
                        wb = wb.astype(self.dtype)
                        self.weight_bc.append(wb)
                             
                #run with the distribute input
                batch_ls,batch_le,batch_ld,batch_lb = distributed_train_step(batch_data, wd, wb)              
                # estimate the mean loss
                epoch_ls_ave(batch_ls)
                epoch_le_ave(batch_le)
                epoch_ld_ave(batch_ld)
                epoch_lb_ave(batch_lb)
                
            # record the loss at every epoch
            ls = epoch_ls_ave.result()
            le = epoch_le_ave.result()
            ld = epoch_ld_ave.result()
            lb = epoch_lb_ave.result()                 
            # record the loss
            self.loss_all.append(ls)
            self.loss_eqns.append(le)
            self.loss_fdat.append(ld)
            self.loss_conds.append(lb)
            # learning rate
            lr = self.optimizer._decayed_lr(tf.float32).numpy()    
            # record the time
            t2 = time.time()
            # output
            print("TensorFlow Epoch %05d-->loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e, learning rate: %8.6f, used time(sec):%4.2f" %
                    (epoch, ls, ld, le, lb, lr, t2-t1))
            
            # update the initial data
            
            # save mode
            if (epoch % 100 == 0 and epoch != 0) or (epoch == epochs-1):
                # save model and parameters
                self.saveNN(index=epoch)
  
    
   
    def train_bfgs(self):
        """
        trained by L-BFGS-B
        """
        self.training = True 
        # generate the dataset
        self.global_batch_size = self.bfgs_batch_size
        self.cond_batch_size = int(self.global_batch_size/5.0)
        batch_size = self.bfgs_batch_size
        train_data, val_data = generate_dataset(self.data_eqns, 
                                                data_rate=self.data_rate,
                                                batchsize=batch_size,
                                                isrepeat=False)
        epochs = self.bfgs_epochs   
        # main loop           
        for epoch in np.arange(0, epochs):
            # record the total loss
            epoch_ls_ave = tf.keras.metrics.Mean()
            # record the equation loss           
            epoch_le_ave = tf.keras.metrics.Mean()
            # record the data loss                
            epoch_ld_ave = tf.keras.metrics.Mean()
            # record the boundary condition loss                
            epoch_lb_ave = tf.keras.metrics.Mean()
            # L-BFGS 
            for batch_X, batch_Y in train_data:                  
                func, params, names = tf_function_factory(self.model, self.loss_fn_all, batch_X[:,0:4], batch_Y)
                # Minimization
                res = minimize(func, 
                                params, 
                                method='L-BFGS-B',
                                options={'disp':None,
                                        'maxiter': 20,
                                        'maxcor': 50,
                                        'maxls': 40,
                                        'gtol':1e-8,
                                        'eps':1e-8,
                                        'ftol': 1e-12})  
                
                # func = function_factory(self.model, self.loss_fn_all, batch_X, batch_Y)

                # # convert initial model parameters to a 1D tf.Tensor
                # init_params = tf.dynamic_stitch(func.idx, self.model.trainable_variables)
                
                # # train the model with L-BFGS solver
                # results = tfp.optimizer.lbfgs_minimize(
                #     value_and_gradients_function=func, initial_position=init_params, max_iterations=10)
                
                # # after training, the final optimized parameters are still in results.position
                # # so we have to manually put them back to the model
                # func.assign_new_model_parameters(results.position)

                # estimate the mean loss
                batch_ls, batch_le, batch_ld, batch_lb = self.loss_custom(batch_X, batch_Y)
                epoch_ls_ave(batch_ls)
                epoch_le_ave(batch_le)
                epoch_ld_ave(batch_ld)
                epoch_lb_ave(batch_lb)
            
            # record the loss at every epoch
            ls = epoch_ls_ave.result()
            self.loss_all.append(ls)
            le = epoch_le_ave.result()
            self.loss_eqns.append(le)
            ld = epoch_ld_ave.result()
            self.loss_fdat.append(ld)
            lb = epoch_lb_ave.result()
            self.loss_conds.append(lb)         
            # output
            print("L-BFGS-B Epoch %05d-->loss: %.4e, loss_data: %.4e, loss_eqns: %.4e, loss_conds: %.4e" %
                    (epoch, ls, ld, le, lb))
            # save mode
            if (epoch % 5 == 0) or (epoch == epochs-1):
                # save model and parameters
                self.saveNN()
                
    
        
    def predict(self, input_X):
        # prediction  
        Y = self.model.predict(input_X, batch_size=10000*self.gpus_number,
                               workers=4, use_multiprocessing=True)
        # Y = self.model(input_X, training=False)
        u = Y[:,0:1]
        v = Y[:,1:2]
        w = Y[:,2:3]
        p = Y[:,3:4]  
        # recover to real value
        u = u*self.norm_paras[1,4]+self.norm_paras[0,4]
        v = v*self.norm_paras[1,5]+self.norm_paras[0,5]
        w = w*self.norm_paras[1,6]+self.norm_paras[0,6]
        p = p*self.norm_paras[1,7]+self.norm_paras[0,7]
        
        return u, v, w, p
    
    
    
    # def loadNN(self):
    #     weight_file = './weights/'+ self.savename
    #     print("load the model: %s" % weight_file)
    #     model = self.build_model()
    #     model.load_weights(weight_file)
    #
    #     return model
    #
    #
    #
    # def restoreNN(self):
    #     check_dir = './weights/'
    #     lastest = tf.train.latest_checkpoint(check_dir)
    #     model = self.build_model()
    #     model.load_weights(lastest)
    #
    #     return model

    def loadNN(self, index='final'):
        # the saved path
        savepath = './weights/' + self.savename + '/'
        if index == 'final':
            savename = self.savename
        else:
            str1 = self.savename
            str2 = '_epoch_%06d' % index
            savename = str1 + str2

        model = self.build_model()
        weight = savepath + savename
        print("load the model: %s" % weight)
        model.load_weights(weight)
        return model

    def restoreNN(self):
        check_dir = './weights/' + self.savename + '/'
        lastest = tf.train.latest_checkpoint(check_dir)
        model = self.build_model()
        model.load_weights(lastest)

        return model

    def saveNN(self, index=0):
        # the save path
        savepath = './weights/'+self.savename+'/'
        str1 = self.savename
        str2 = '_epoch_%06d' % index
        # the save name
        savename = str1+str2
        # make dir
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        # save the model with epoch
        print("save the model and paras: %s" % savepath)
        # save the network
        self.model.save_weights(savepath+savename)
        # save the parameters
        sio.savemat(savepath+savename+'_paras.mat', 
                    {'batch_size':self.tf_batch_size,
                     'epochs':index,
                     'loss_all':self.loss_all,
                     'loss_fdat':self.loss_fdat,   
                     'loss_eqns':self.loss_eqns,
                     'loss_conds':self.loss_conds,
                     'weight_fdat':self.weight_fdat,
                     'weight_bc':self.weight_bc,
                     'norm_paras':self.norm_paras})   
        # save the model without epoch
        savename = str1
        # save the network
        self.model.save_weights(savepath+savename)
        # save the parameters
        sio.savemat(savepath+savename+'_paras.mat', 
                    {'batch_size':self.tf_batch_size,
                     'epochs':index,
                     'loss_all':self.loss_all,
                     'loss_fdat':self.loss_fdat,   
                     'loss_eqns':self.loss_eqns,
                     'loss_conds':self.loss_conds,
                     'weight_fdat':self.weight_fdat,
                     'weight_bc':self.weight_bc,
                     'norm_paras':self.norm_paras})   
   
    
             
    def loadParas(self, index='final'):
        """
        Returns
        -------
        norm_paras
        loadmat: no transpose
        """
        # the saved path
        savepath = './weights/'+self.savename+'/'
        if index == 'final':
            savename = self.savename
        else:
            str1 = self.savename
            str2 = '_epoch_%06d' % index
            savename = str1 + str2
            
        print("load the paras: %s" % savepath+savename+'_paras.mat')
        data = sio.loadmat(savepath+savename+'_paras.mat')
        self.norm_paras = data['norm_paras']
        # load the weight data
        weight_fdat = data['weight_fdat']
        weight_bc = data['weight_bc']
        # weight is a two-dimensional array, convert array list[array,array,...]
        self.weight_fdat = [np.array(i) for i in weight_fdat[-1]]
        self.weight_bc = [np.array(i) for i in weight_bc[-1]]
        # load the loss data
        loss_all = data['loss_all']
        loss_fdat = data['loss_fdat']
        loss_eqns = data['loss_eqns']
        loss_conds = data['loss_conds']
        # loss is a two-dimensional array, convert array list[array,array,...]
        self.loss_all = [np.array(i) for i in loss_all[-1]]
        self.loss_fdat = [np.array(i) for i in loss_fdat[-1]] 
        self.loss_eqns = [np.array(i) for i in loss_eqns[-1]]           
        self.loss_conds = [np.array(i) for i in loss_conds[-1]] 
        # convert to scalar
        self.epochs_last = data['epochs'].item()
        
    

    def piecewise_scheduler(self, epoch):
        """
        piecewise learning rate decay
        """
        rate = np.floor(epoch/100)
        return self.tf_config.init_lr/(rate+1.0)
    
    

    def exponential_continuous_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 10
        lr = self.tf_config.init_lr * np.power(decay_rate,(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr 
        
        
        
        
    def exponential_staircase_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.98
        decay_epoch = 10
        lr = self.tf_config.init_lr * np.power(decay_rate, np.floor(epoch / decay_epoch))
        if lr < 1e-6:
            return 1e-6
        else:
            return lr        
            
    


    def constant_scheduler(self, epoch):
        """
        constant learning rate decay
        """
        return self.tf_config.init_lr   

    
