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

 
    
 
    
class NS2D_UnSteady_PINNs(object):
    def __init__(self, paras):
        """
        hp: parameters            
        Note: equation points should include the points of boundary conditions
        """
        # clear session
        K.clear_session()
        # hp is the structure of hyper-parameters
        self.dtype = 'float32'
        # layer_f is the neural network for fluid Euler coordinates
        self.layer = paras['layers']
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
        self.data_flow = None
        self.data_neuralbc = None
        
        # Initialize the loss recording list
        self.loss_all = []
        self.loss_fdat = []
        self.loss_eqns = []
        self.loss_conds = []
        # weight from gradient statistics; at epoch=0 the flow-field weight is increased
        self.weight_fdat = [np.array(1.0)]
        self.weight_bc = [np.array(1.0)]
        

        # Multi GPU or single GPU
        if _GPU_NUM > 1:
            # tf.distribute.NcclAllReduce() only for linux
            # self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
            self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
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
        reduce_lr = MyLRSchedule(self.init_lr, 1.0e-5, self.tf_epochs, self.epochs_last, 50, self.steps_per_epoch, 0.95)

        # generate the model and optimizer within strategy
        if self.training:
            with self.strategy.scope():
                # optimizer
                self.optimizer = Adam(reduce_lr)  
                if self.ExistModel == 0:
                    self.norm_paras = paras['norm_paras']
                    self.model = self.build_model()
                    
                elif self.ExistModel == 1:
                    self.model = self.loadNN(index='final')
        else:
            # optimizer
            self.optimizer = Adam(reduce_lr)  
            if self.ExistModel == 0:
                self.norm_paras = paras['norm_paras']
                self.model = self.build_model()
                
            elif self.ExistModel == 1:
                self.model = self.loadNN(index='final')          
                    
        print("The neural network for solving the N-S equation:")            
        self.model.summary()          
        

    
    
    def build_model(self):
        # generate neural network for fluid
        model = generator(self.layer, self.norm_paras, maptype=self.maptype)
        return model
    
 
    
    def init_conditions(self):
        """
        initilize the condition dictionary
        """
        conds_keys = ['init', 'uv',
                      'u',  'v', 'p',
                      'ux', 'uy',
                      'vx', 'vy',
                      'px', 'py']
        
        # default is no conditions
        conds_dict = dict.fromkeys(conds_keys, None)
       
        return conds_dict
            
 
            

    def set_cond_bc(self, keys, X=None):
        """
        set the boundary
        note:  x,yis the locations
        initial data: X=[t,x,y,u,v,p]
        Dirichlet BC of uv: X=[t,x,y,u,v]
        Dirichlet BC of X: X = [t,x,y,x]       
        """    
        key = keys.lower()
        if X is None:
            self.conds[key] = None
        else:
            train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                    data_rate=1.0,
                                                    batchsize=self.cond_batch_size,
                                                    isrepeat=True)
            # distribute the dataset over replicas
            train_data = self.strategy.experimental_distribute_dataset(train_data)
            val_data = self.strategy.experimental_distribute_dataset(val_data)
            # use iter to create the iterator            
            self.conds[key]  = [train_data, iter(train_data)]
  
     


    def set_eqns_points(self, X=None):
        """
        set the data points for equations
        data equations: X=[t,x,y]
        """ 
        # generate the dataset
        if X is None:
            self.data_eqns = None
        else:
            train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                    data_rate=self.data_rate,
                                                    batchsize=self.global_batch_size,
                                                    isrepeat=True)
            # distribute the dataset over replicas
            train_data = self.strategy.experimental_distribute_dataset(train_data)
            val_data = self.strategy.experimental_distribute_dataset(val_data) 
            # estimate the batch number
            # self.batch_num = int(X.shape[0]*self.data_rate/self.global_batch_size)
            # self.batch_num = 30
            # use iter to create the iterator
            self.data_eqns = [train_data, iter(train_data)]

 


    def set_supervised_flow(self, X=None):
        """
        set the flow data points for supervising
        data supervised: X=[t,x,y,u,v,p]
        """   
        if X is None:
            self.data_flow = None
        else:
             train_data, val_data = generate_dataset(X.astype(self.dtype), 
                                                     data_rate=1.0,
                                                     batchsize=self.cond_batch_size,
                                                     isrepeat=True)
             # distribute the dataset over replicas
             train_data = self.strategy.experimental_distribute_dataset(train_data)
             val_data = self.strategy.experimental_distribute_dataset(val_data)
             # use iter to create the iterator
             self.data_flow = [train_data, iter(train_data)]
             

           
                
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
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint             
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            tape2.watch(x)
            tape2.watch(y)
                
            with tf.GradientTape(persistent=True) as tape1:
                # Watching gradients of t,x,y,z
                tape1.watch(t)
                tape1.watch(x)
                tape1.watch(y)
                # Packing together the inputs
                X = tf.stack([t[:,0],x[:,0],y[:,0]], axis=1)
                # Getting the prediction
                # Y = self.model(X, training=self.training)
                # Y: [u,v,p,um,vm,pm,ustd,vstd,pstd]
                Y = self.model(X)
                u = Y[:,0:1]
                v = Y[:,1:2]
                p = Y[:,2:3]
                # recover to real value
                u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
                v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
                p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
            # first-order deriavative
            u_t = tape1.gradient(u, t)
            v_t = tape1.gradient(v, t)
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)

        # second-order deriavative
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        
        e1 = u_t + (u * u_x + v * u_y ) + p_x - (1.0 / self.Re) * (u_xx + u_yy )
        e2 = v_t + (u * v_x + v * v_y ) + p_y - (1.0 / self.Re) * (v_xx + v_yy )
        e3 = u_x + v_y 
        
        # # cylindrical coordinates
        # e1 = u_t + (u * u_x + v * u_y ) + p_x - (1.0 / self.Re) * (u_xx + u_yy + 1.0 / y * u_y )
        # e2 = v_t + (u * v_x + v * v_y ) + p_y - (1.0 / self.Re) * (v_xx + v_yy + 1.0 / y * v_y - v / y / y)
        # e3 = u_x + v_y + v / y

        # e1 = 0.0 * e1
        # e2 = 0.0 * e2
        # e3 = 0.0 * e3

        # Letting the tape go
        del tape1, tape2
        # Buidling the PINNs
        return e1, e2, e3
    

  
    @tf.function
    def get_gradient(self, X, flag):
        """
        Returns
        -------
        the gradient of u,v,p

        """    
        keys = ['ux', 'uy',
                'vx', 'vy',
                'px', 'py']
        # default is no conditions
        grad = dict.fromkeys(keys, None)
        
        t = tf.convert_to_tensor(X[:,0:1], self.dtype)
        x = tf.convert_to_tensor(X[:,1:2], self.dtype)
        y = tf.convert_to_tensor(X[:,2:3], self.dtype)
        
        # Using the new GradientTape paradigm of TF2.0
        # persistent: multi-times gradint               
        with tf.GradientTape(persistent=True) as tape:
            # Watching gradients of t,x,y,z
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            # Packing together the inputs
            X = tf.stack([t[:,0],x[:,0],y[:,0]], axis=1)
            # Getting the prediction
            # Y = self.model(X, training=self.training)
            Y = self.model(X)
            u = Y[:,0:1]
            v = Y[:,1:2]
            p = Y[:,2:3]
            # recover to real value
            u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
            v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
            p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
        # first-order deriavative
        if flag.lower() == 'ux':
            grad['ux'] = tape.gradient(u, x)
        elif flag.lower() == 'uy':      
            grad['uy'] = tape.gradient(u, y)
        elif flag.lower() == 'vx':
            grad['vx'] = tape.gradient(v, x)
        elif flag.lower() == 'vy':
            grad['vy'] = tape.gradient(v, y)        
        elif flag.lower() == 'px':
            grad['px'] = tape.gradient(p, x)  
        elif flag.lower() == 'py':
            grad['py'] = tape.gradient(p, y)  
        elif flag.lower() == 'all':
            grad['ux'] = tape.gradient(u, x)
            grad['uy'] = tape.gradient(u, y)
            grad['vx'] = tape.gradient(v, x)
            grad['vy'] = tape.gradient(v, y)       
            grad['px'] = tape.gradient(p, x)    
            grad['py'] = tape.gradient(p, y)  
           
    
        # Letting the tape go
        del tape
        
        # Buidling the PINNs
        return grad
    
    
    
    @tf.function
    def get_uvp(self, X):
        """
        get the output of the network
        The predict function is designed for performance in large scale inputs. 
        For small amount of inputs that fit in one batch, directly using
        __call__() is recommended for faster execution, e.g., model(x), 
        or model(x, training=False)

        """ 
        Xi = tf.convert_to_tensor(X[:,0:3], self.dtype)
        Y = self.model(Xi)
        u = Y[:,0:1]*self.norm_paras[1,3]+self.norm_paras[0,3]
        v = Y[:,1:2]*self.norm_paras[1,4]+self.norm_paras[0,4]
        p = Y[:,2:3]*self.norm_paras[1,5]+self.norm_paras[0,5]
        
        return u, v, p   
        
    
    
    
    def get_dynamic_weight(self, batch_data):
        """
        Compute the dynamic loss weights.
        Gradients are computed separately to save memory.
        """
        with tf.GradientTape(persistent=False) as tap:
            # loss of equation
            le = self.loss_fn_eqns(batch_data)
        # gradient of the equation loss w.r.t. the network parameters
        ge = tap.gradient(le,
                         self.model.trainable_variables,
                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
        ge = [tf.reshape(j, (1, -1)) for j in ge]
        ge = tf.concat(ge, axis=1)
        # ge_mean = tf.math.reduce_mean(tf.abs(ge))
        ge_mean = tf.math.reduce_mean(tf.abs(ge))/tf.sqrt(le)
        #     ge_list(tf.reduce_mean(tf.abs(tmp)))
        del tap
        
        with tf.GradientTape(persistent=False) as tap:
            # loss of data
            ld = self.loss_flow_data(batch_data)    
        # gradient of the data loss w.r.t. the network parameters
        gd = tap.gradient(ld,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gd = [tf.reshape(j, (1, -1)) for j in gd]
        gd = tf.concat(gd, axis=1)
        # gd_mean = tf.math.reduce_mean(tf.abs(gd))
        gd_mean = tf.math.reduce_mean(tf.abs(gd))/tf.sqrt(ld)
        # for tmp in gd:
        #     gd_list(tf.reduce_mean(tf.abs(tmp)))
        del tap
        
        with tf.GradientTape(persistent=False) as tap:
            # loss of boundary condtions
            lb = self.loss_fn_conds(batch_data)
        # gradient of the boundary loss w.r.t. the network parameters
        gb = tap.gradient(lb,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)

        gb = [tf.reshape(j, (1, -1)) for j in gb]
        gb = tf.concat(gb, axis=1)
        # gb_mean = tf.math.reduce_mean(tf.abs(gb))
        gb_mean = tf.math.reduce_mean(tf.abs(gb))/tf.sqrt(lb)
        del tap
        
        # weights from gradient statistics, biased toward the data
        wd = tf.minimum(ge_mean/gd_mean,1.0e3)
        wb = tf.minimum(ge_mean/gb_mean,1.0e3)
        # wb = tf.zeros_like(wd)
        
        return wd, wb
    
    
    
    @tf.function
    def grad_custom(self, batch_data, weight_fdat, weight_bc):
        """
        Compute the gradient of the total loss for inputs X=[t,x,y].
        """ 
        self.iternum = self.iternum+1          
        # estimate the total loss
        with tf.GradientTape(persistent=False) as tap:

            # loss of equation
            le = self.loss_fn_eqns(batch_data)
            # loss of data
            ld = self.loss_flow_data(batch_data)    
            # loss of boundary condtions
            lb = self.loss_fn_conds(batch_data)
            # the dynamic weights are updated before calling this function
            ls = le + weight_fdat*ld + weight_bc*lb
            # ls = le + weight_bc * lb
 
        g = tap.gradient(ls,
                          self.model.trainable_variables,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        del tap
        return g, ls, le, ld, lb
 
    
 
    def loss_fn_eqns(self, data):
        """
        return the loss of equations
        """   
        Y_true = data['eqns'][0]
        # [t,x,y,ws,wt]
        Y_true = Y_true[:,0:5]
        # first compute the loss at the equation points
        # Y_true is the same as X
        X = Y_true[:,0:3]
        # the weight for space
        ws = Y_true[:,3:4]
        # the weight for time
        wt = Y_true[:,4:5]
        e1,e2,e3 = self.ns_eqns(X)          
        # e1 = e1 * ws * wt
        # e2 = e2 * ws * wt
        # e3 = e3 * ws * wt
        
        # multi-gpu: sum divided by global_batch_size
        loss_eqns_e1 = tf.reduce_sum(tf.square(e1))*(1.0/self.global_batch_size)
        loss_eqns_e2 = tf.reduce_sum(tf.square(e2))*(1.0/self.global_batch_size)
        loss_eqns_e3 = tf.reduce_sum(tf.square(e3))*(1.0/self.global_batch_size)
        loss_eqns = loss_eqns_e1+loss_eqns_e2+loss_eqns_e3

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
            if (key == 'init') or (key == 'uv'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:3]
                    ut = batch_x[:,3:4]
                    vt = batch_x[:,4:5]
                    pt = batch_x[:,5:6]
                    # pt = val[idx,5:6]
                    up, vp, pp = self.get_uvp(tmpX)
                    
                    # set the weight according to the error magnitude
                    tmpu = tf.square(ut-up)
                    tmpv = tf.square(vt-vp)
                    # multi-gpu: sum divided by global_batch_size
                    lossu = tf.reduce_sum(tmpu)*(1.0/self.cond_batch_size)
                    lossv = tf.reduce_sum(tmpv)*(1.0/self.cond_batch_size)   
                    
                    loss = loss + 1.0*(lossu + lossv)
                                
                    
            elif (key == 'u'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:3]
                    ut = batch_x[:,3:4]
                    # pt = val[idx,5:6]
                    up, vp, pp = self.get_uvp(tmpX)
                    # set the weight according to the error magnitude
                    tmp = tf.square(ut-up)
                    # multi-gpu: sum divided by global_batch_size
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
                    tmpX = batch_x[:,0:3]
                    vt = batch_x[:,3:4]
                    # pt = val[idx,5:6]
                    up, vp, pp = self.get_uvp(tmpX)
                    # set the weight according to the error magnitude
                    tmp = tf.square(vt-vp)      
                    # multi-gpu: sum divided by global_batch_size
                    lossv = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size)
                    loss = loss + lossv   
 

            elif (key == 'p'):
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:3]
                    pt = batch_x[:,3:4]
                    # pt = val[idx,5:6]
                    up, vp, pp = self.get_uvp(tmpX)
                    # set the weight according to the error magnitude
                    tmp = tf.square(pt-pp)
                    # multi-gpu: sum divided by global_batch_size
                    lossp = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size)
                    loss = loss + lossp                    
                    
            # for the conditions of gradients: ux, uy, vx, vy, px, py
            else:
                if val is None:
                    loss = loss+0.0
                else:
                    # a small batch to estimate the loss
                    batch_x = data[key][0]
                    batch_y = data[key][1]
                    # using the fluctuation to calculate the error
                    tmpX = batch_x[:,0:3]
                    gt = batch_x[:,3:4]
                    gp = self.get_gradient(tmpX, key)
                    # set the weight according to the error magnitude
                    tmp = tf.square(gt-gp[key])
                    # multi-gpu: sum divided by global_batch_size
                    tmp = tf.reduce_sum(tmp)*(1.0/self.cond_batch_size)
                    loss = loss + tmp      
                    # # Debug output: Print key and loss contribution
                    # tf.print("Debug: key =", key, ", loss contribution =", tmp)
        return loss
    


    def loss_flow_data(self, data):
        """
        return the loss of supervised data
        """     
        if self.data_flow is None:
            loss = 0.0
        else:
            x = data['data_flow'][0]
            y = data['data_flow'][1]
            # using the fluctuation to calculate the error
            tmpX = x[:,0:6]           
            # prediction
            up, vp, pp = self.get_uvp(tmpX[:,0:3])
            
            # get the data value
            ut = tmpX[:,3:4]
            vt = tmpX[:,4:5]
            #pt = tmpX[:,4:5]
    
            # multi-gpu: sum divided by global_batch_size
            lossu = tf.reduce_sum(tf.square(ut-up))*(1.0/self.cond_batch_size)
            lossv = tf.reduce_sum(tf.square(vt-vp))*(1.0/self.cond_batch_size)
            
            loss = lossu+lossv # 10x weight was added to lossv for the circular-pipe flow
        return loss                 
  


    def get_batch_data(self):
        """
        record the batch data for distribute training
        """
        batch_keys = ['eqns','data_flow',
                      'init', 'uv',
                      'u',  'v', 'p',
                      'ux', 'uy',
                      'vx', 'vy',
                      'px', 'py']
        
        # default is no conditions
        batch_dict = dict.fromkeys(batch_keys, None)
        # record
        if self.data_eqns is not None:
            it = self.data_eqns[1]
            batch_x, batch_y = next(it)
            batch_dict['eqns'] = [batch_x, batch_y]
            
        if self.data_flow is not None:
            it = self.data_flow[1]
            batch_x, batch_y = next(it)
            batch_dict['data_flow'] = [batch_x, batch_y]
            
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

        ############################################################################################
        def train_step(batch_data, wd, wb):
            # wd is the weight of data
            # wb is the weight of BC
            # estimate the loss and gradient
            batch_grad, batch_ls, batch_le, batch_ld, batch_lb = self.grad_custom(batch_data, wd, wb)
            # gradient optimize
            self.optimizer.apply_gradients(zip(batch_grad, self.model.trainable_variables))
            return batch_ls, batch_le, batch_ld, batch_lb
        ##############################################################################################
                     
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
                # first converge the moving-boundary network
                if count == 0:
                    # estimate the dynamic weights (stored in self.weight_data and self.weight_bc)
                    wd, wb = self.strategy.run(self.get_dynamic_weight, args=(batch_data,))
                    # average across GPUs
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
                        # wd = np.array(1.0)
                        # wd = wd.astype(self.dtype)
                        # wb = np.array(1.0)
                        # wb = wd.astype(self.dtype)
                        # wm = np.array(1.0)
                        # wm = wd.astype(self.dtype)
                        # record the weight: alpha = (1-lambda)*alpha+lambda*alpha
                        wd = (1-0.1)*self.weight_fdat[-1]+0.1*K.get_value(wd)
                        wd = wd.astype(self.dtype)
                        self.weight_fdat.append(wd)
                        wb = (1-0.1)*self.weight_bc[-1]+0.1*K.get_value(wb)
                        wb = wb.astype(self.dtype)
                        self.weight_bc.append(wb)                                     
                        
                # run with the distribute input
                batch_ls, batch_le, batch_ld, batch_lb = distributed_train_step(batch_data, wd, wb)    
                # estimate the mean loss
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
                
    
        
    def predict_field(self, input_X):
        # prediction  
        Y = self.model.predict(input_X, batch_size=10000*self.gpus_number,
                               workers=4, use_multiprocessing=True)
        # Y = self.model(input_X, training=False)
        u = Y[:,0:1]
        v = Y[:,1:2]
        p = Y[:,2:3]
        # recover to real value
        u = u*self.norm_paras[1,3]+self.norm_paras[0,3]
        v = v*self.norm_paras[1,4]+self.norm_paras[0,4]
        p = p*self.norm_paras[1,5]+self.norm_paras[0,5]
        
        return u, v, p
            
     
    
    def loadNN(self, index='final'):
        # the saved path
        savepath = './weights/'+self.savename+'/'
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
        check_dir = './weights/'+self.savename+'/'
        lastest = tf.train.latest_checkpoint(check_dir)
        model= self.build_model()
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

    
