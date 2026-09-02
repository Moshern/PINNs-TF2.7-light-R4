# -*- coding: utf-8 -*-
# Copyright (c) 2020 Fabio Di Marco
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""
import sympy as sp
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot





# ==============================================================================
def generate_dataset(data, data_rate=1.0, batchsize=64, isrepeat=True):
    """
    generate dataset
    """
    # note: always has equation point
    # only consider equation point in train
    # train_Y is equal to train_X to record the coordinates
    data_num = data.shape[0]
    # 对于Dataset，batch_size越大越好
    # 当数据较小时，全部用于训练
    batch_size = np.minimum(batchsize,int(data_num*data_rate))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # 取出数据
    train_data = tf.data.Dataset.from_tensor_slices((data,data)).take(int(data_num*data_rate))
    train_data = train_data.with_options(options)
    # 打乱数据
    train_data = train_data.shuffle(buffer_size=np.floor(data_num/1.0))
    if isrepeat:
        # The default behavior (if count is None or -1) is for the dataset be repeated indefinitely.
        train_data = train_data.repeat(None)
    # set the batch
    train_data = train_data.batch(batch_size)
    # 预处理，CPU提前准备数据
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 跳过数据
    val_data = tf.data.Dataset.from_tensor_slices((data,data)).skip(int(data_num*data_rate))
    val_data = val_data.with_options(options)
    val_data = val_data.batch(batch_size) 
    # 预处理，CPU提前准备数据
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_data, val_data




# ==============================================================================
@tf.function
def jaco_custom(model, X, Y): 
    eps = tf.keras.backend.epsilon()
    with tf.GradientTape(persistent=True) as tap:
        # predicted by the model
        Y_pred = model(X, training=True)
        ls = tf.square(Y_pred-Y)
        # residual
        res = tf.math.sqrt(eps+ls)
    jaco = tap.jacobian(res,
                    model.trainable_variables,
                    experimental_use_pfor=True,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)
    loss = tf.reduce_mean(ls)
    del tap
    return jaco, res, loss


@tf.function
def loss_custom(model, X, Y):
    # predicted by the model
    Y_pred = model(X, training=False)
    loss = tf.reduce_mean(tf.square(Y_pred-Y))
    # 注意：有可能会有多个返回值
    # 返回list, toal_loss放在第一位
    return [loss]




# ==============================================================================

class DampingAlgorithm:
    """Default Levenberg–Marquardt damping algorithm.

    This is used inside the Trainer as a generic class. Many damping algorithms
    can be implemented using the same interface.
    """

    def __init__(self,
                 starting_value=1e-2,
                 dec_factor=0.1,
                 inc_factor=10.0,
                 min_value=1e-12,
                 max_value=1e+12,
                 adaptive_scaling=False,
                 fletcher=False):
        """Initializes `DampingAlgorithm` instance.

        Args:
          starting_value: (Optional) Used to initialize the Trainer internal
            damping_factor.
          dec_factor: (Optional) Used in the train_step decrease the
            damping_factor when new_loss < loss.
          inc_factor: (Optional) Used in the train_step increase the
            damping_factor when new_loss >= loss.
          min_value: (Optional) Used as a lower bound for the damping_factor.
            Higher values improve numerical stability in the resolution of the
            linear system, at the cost of slower convergence.
          max_value: (Optional) Used as an upper bound for the damping_factor,
            and as condition to stop the Training process.
          adaptive_scaling: Bool (Optional) Scales the damping_factor adaptively
            multiplying it with max(diagonal(JJ)).
          fletcher: Bool (Optional) Replace the identity matrix with
            diagonal of the gauss-newton hessian approximation, so that there is
            larger movement along the directions where the gradient is smaller.
            This avoids slow convergence in the direction of small gradient.
        """
        self.starting_value = starting_value
        self.dec_factor = dec_factor
        self.inc_factor = inc_factor
        self.min_value = min_value
        self.max_value = max_value
        self.adaptive_scaling = adaptive_scaling
        self.fletcher = fletcher

    def init_step(self, damping_factor, loss):
        return damping_factor

    def decrease(self, damping_factor, loss):
        return tf.math.maximum(
            damping_factor * self.dec_factor,
            self.min_value)

    def increase(self, damping_factor, loss):
        return tf.math.minimum(
            damping_factor * self.inc_factor,
            self.max_value)

    def stop_training(self, damping_factor, loss):
        return damping_factor >= self.max_value

    def apply(self, damping_factor, JJ):
        if self.fletcher:
            damping = tf.linalg.tensor_diag(tf.linalg.diag_part(JJ))
        else:
            damping = tf.eye(tf.shape(JJ)[0], dtype=JJ.dtype)

        scaler = 1.0
        if self.adaptive_scaling:
            scaler = tf.math.reduce_max(tf.linalg.diag_part(JJ))

        damping = tf.scalar_mul(scaler * damping_factor, damping)
        return tf.add(JJ, damping)

# ==============================================================================






class LevenbergMarquardt(object):
    def __init__(self, 
               model, 
               jaco_func=jaco_custom,
               loss_func=loss_custom,
               optimizer=keras.optimizers.SGD(learning_rate=1.0),
               solve_method='solve',
               damping_algorithm=DampingAlgorithm(),
               attempts_per_step=10,
               jacobian_max_num_rows=100):
        """
        model : It is the Model to be trained, it is expected to inherit
                from tf.keras.Model and to be already built.
        jaco_func: return the Jacobian function and residuals
        loss_func: return the loss function
        optimizer: (Optional) Performs the update of the model trainable
          variables. When tf.keras.optimizers.SGD is used it is equivalent
          to the operation `w = w - learning_rate * updates`, where updates is
          the step computed using the Levenberg-Marquardt algorithm.
        solve_method: (Optional) Possible values are:
          'qr': Uses QR decomposition which is robust but slower.
          'cholesky': Uses Cholesky decomposition which is fast but may fail
              when the hessian approximation is ill-conditioned.
          'solve': Uses tf.linalg.solve. I don't know what algorithm it
              implements. But it seems a compromise in terms of speed and
              robustness.
        damping_algorithm: (Optional) Class implementing the damping
          algorithm to use during training.      
        jacobian_max_num_rows: Integer (Optional) When the number of residuals
          is greater then the number of variables (overdetermined), the
          hessian approximation is computed by slicing the input and
          accumulate the result of each computation. In this way it is
          possible to drastically reduce the memory usage and increase the
          speed as well. The input is sliced into blocks of size less than or
          equal to the jacobian_max_num_rows.

        """
        self.model = model
        self.jaco_func = jaco_func
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.jacobian_max_num_rows = jacobian_max_num_rows
        self.attempts_per_step = attempts_per_step
        self.damping_algorithm = damping_algorithm
        self.dtype = 'float32'
        self.old_loss = 0.0
        self.new_loss = 0.0
        
        # Define and select linear system equation solver.
        def qr(matrix, rhs):
            q, r = tf.linalg.qr(matrix, full_matrices=True)
            y = tf.linalg.matmul(q, rhs, transpose_a=True)
            return tf.linalg.triangular_solve(r, y, lower=False)

        def cholesky(matrix, rhs):
            chol = tf.linalg.cholesky(matrix)
            return tf.linalg.cholesky_solve(chol, rhs)

        def solve(matrix, rhs):
            return tf.linalg.solve(matrix, rhs)

        if solve_method == 'qr':
            self.solve_function = qr
        elif solve_method == 'cholesky':
            self.solve_function = cholesky
        elif solve_method == 'solve':
            self.solve_function = solve
        else:
            raise ValueError('Invalid solve_method.')

        # Keep track of the current damping_factor.
        self.damping_factor = tf.Variable(
            self.damping_algorithm.starting_value,
            trainable=False,
            dtype=self.dtype)
        
        # Used to backup and restore model variables.
        self._backup_variables = []

        # Since training updates are computed with shape (num_variables, 1),
        # self._splits and self._shapes are needed to split and reshape the
        # updates so that they can be applied to the model trainable_variables.
        self._splits = []
        self._shapes = []

        for variable in self.model.trainable_variables:
            variable_shape = tf.shape(variable)
            variable_size = tf.reduce_prod(variable_shape)
            backup_variable = tf.Variable(
                tf.zeros_like(variable),
                trainable=False)

            self._backup_variables.append(backup_variable)
            self._splits.append(variable_size)
            self._shapes.append(variable_shape)
    


    # def trans_Jac_Res(self, Grad_list, Res_list):
    #     """
    #     Jac_list: list of Jacobian Array
    #     Res_list: list of residual Array
    #     Returns
    #     -------
    #     None.
    
    #     """
    #     num_residuals = len(Res_list)
    #     num_variables = tf.reduce_prod(tf.shape(Grad_list[0]))
        
    #     residuals = tf.concat(Res_list, axis=0)
    #     residuals = tf.reshape(residuals, (num_residuals, -1))
        
    #     jacobian = tf.concat(Grad_list, axis=0)
    #     jacobian = tf.reshape(jacobian, (num_residuals, -1))
    #     self.residuals = residuals
    #     self.jacobian = jacobian
    #     self.num_residuals = num_residuals
    #     self.num_variables = num_variables
    #     self.update_computed = False
    def compute_Jac_Res(self, inputs, targets):
        # compute the jacobian and residuals 
        Jaco, Res, loss = self.jaco_func(inputs, targets)
        self.old_loss = loss
        # tranform the array
        num_residuals = tf.reduce_prod(tf.shape(Res))
        residuals = tf.reshape(Res,(num_residuals, -1))
        
        jacobians = [tf.reshape(j, (num_residuals, -1)) for j in Jaco]
        jacobians = tf.concat(jacobians, axis=1)
        num_variables = tf.shape(jacobians)[1]
        
        self.residuals = residuals
        self.jacobian = jacobians
        self.num_residuals = num_residuals
        self.num_variables = num_variables
        self.update_computed = False
        
        

    def _init_gauss_newton_overdetermined(self):
        # num_residuals > num_variables
        # But reduce memory usage by slicing the inputs so that the jacobian
        # matrix will have maximum shape (jacobian_max_num_rows, num_variables)
        # instead of (batch_size, num_variables).
        slice_size = self.jacobian_max_num_rows
        batch_size = self.num_residuals
        num_slices = batch_size // slice_size
        remainder = batch_size % slice_size

        JJ = tf.zeros(
            [self.num_variables, self.num_variables],
            dtype=self.dtype)

        rhs = tf.zeros(
            [self.num_variables, 1],
            dtype=self.dtype)

        for i in tf.range(num_slices):
            J = self.jacobian[i*slice_size:(i+1)*slice_size, :]
            R = self.residuals[i*slice_size:(i+1)*slice_size]
 
            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, R, transpose_a=True)

        if remainder > 0:
            J = self.jacobian[num_slices * slice_size::, :]
            R = self.residuals[num_slices * slice_size::]

            JJ += tf.linalg.matmul(J, J, transpose_a=True)
            rhs += tf.linalg.matmul(J, R, transpose_a=True)

        return JJ, rhs
    
    

    def _init_gauss_newton_underdetermined(self):
        JJ = tf.linalg.matmul(self.jacobian, self.jacobian, transpose_b=True)
        rhs = self.residuals
        return JJ, rhs



    def _compute_gauss_newton_overdetermined(self, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        return updates
    
    

    def _compute_gauss_newton_underdetermined(self, JJ, rhs):
        updates = self.solve_function(JJ, rhs)
        updates = tf.linalg.matmul(self.jacobian, updates, transpose_a=True)
        return updates




    def get_updates(self):
        """
        """
        overdetermined = self.num_residuals >= self.num_variables
        if overdetermined:
            init_gauss_newton = self._init_gauss_newton_overdetermined
            compute_gauss_newton = self._compute_gauss_newton_overdetermined
        else:
            init_gauss_newton = self._init_gauss_newton_underdetermined
            compute_gauss_newton = self._compute_gauss_newton_underdetermined
            
        # Perform normalization for numerical stability.
        normalization_factor = 1.0 / tf.dtypes.cast(
            self.num_residuals,
            dtype=self.model.dtype)
        
        damping_factor = self.damping_algorithm.init_step(
            self.damping_factor, 0.0)
        JJ, rhs = init_gauss_newton()
        JJ *= normalization_factor
        rhs *= normalization_factor
        JJ_damped = self.damping_algorithm.apply(damping_factor, JJ)
        # Compute the updates:
        # overdetermined: updates = (J'*J + damping)^-1*J'*residuals
        # underdetermined: updates = J'*(J*J' + damping)^-1*residuals
        updates = compute_gauss_newton(JJ_damped, rhs)  
        
        return updates
    
    
    
    def update_variables(self, updates):
        """
        """
        if tf.reduce_all(tf.math.is_finite(updates)):
            self.update_computed = True
            # back up the weights
            self.backup_variables()
            # Split and Reshape the updates
            updates = tf.split(tf.squeeze(updates, axis=-1), self._splits)
            updates = [tf.reshape(update, shape)
                       for update, shape in zip(updates, self._shapes)]

            # Apply the updates to the model trainable_variables.
            self.optimizer.apply_gradients(
                zip(updates, self.model.trainable_variables))
            
            
            
    def update_damping_factor(self):
        """
        """
        old_loss = self.old_loss
        new_loss = self.new_loss
        stop_training = False
        if self.update_computed:
            #如果参数已经更新，判断更新前后loss的大小
            if new_loss < old_loss:
                # Accept the new model variables and backup them.
                self.old_loss = self.new_loss
                self.new_loss = self.new_loss
                self.damping_factor = self.damping_algorithm.decrease(
                    self.damping_factor, self.new_loss)
                self.backup_variables()
                #loss减小，不用训练了
                stop_training = True
            else:
                # Restore the old variables and try a new damping_factor.
                self.restore_variables()
                self.damping_factor = self.damping_algorithm.increase(
                    self.damping_factor, new_loss)
        
        else:
            self.damping_factor = self.damping_algorithm.increase(
                self.damping_factor, new_loss)
        
        stop_training2 = self.damping_algorithm.stop_training(
            self.damping_factor, new_loss)    
        
        return (stop_training or stop_training2)
    
    
    
    def train_step(self, inputs, targets):
        # compute the jacobian and residual
        self.compute_Jac_Res(inputs, targets)
        attempt = 0
        attempts = tf.constant(self.attempts_per_step, dtype=tf.int32)
        while tf.constant(True, dtype=tf.bool):     
            #计算更新量
            updates = self.get_updates()
            #更新权重
            self.update_variables(updates)
            #计算loss,如果返回多个值,取第一个
            tmp = self.loss_func(inputs, targets)
            self.new_loss = tmp[0]
            #更新阻尼系数
            stop_training = self.update_damping_factor()
            #print(lm.damping_factor)
            if stop_training:
                break                
            if attempt < attempts:
                attempt += 1
            else:
                break
        # 当new_loss<old_loss, old_loss已经被更新
        return self.old_loss
 
            
            
    def reset_damping_factor(self):
        self.damping_factor.assign(self.damping_algorithm.starting_value)



    def backup_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            backup.assign(variable)



    def restore_variables(self):
        zip_args = (self.model.trainable_variables, self._backup_variables)
        for variable, backup in zip(*zip_args):
            variable.assign(backup)            




def flow2D(tvec, xvec, yvec):
    """
    Parameters
    ----------
    tvec : time 
    xvec : x nodes
    yvec : y nodes

    Returns
    -------
    u, v, p
    """
    x, y, t = sp.symbols('x y t')
    u = -sp.cos(x)*sp.sin(y)*sp.exp(-2.0*t)
    u_f = sp.lambdify([x, y, t], u, 'numpy')
    v = sp.sin(x)*sp.cos(y)*sp.exp(-2.0*t)
    v_f = sp.lambdify([x, y, t], v, 'numpy')    
    p = -0.25*(sp.cos(2.0*x)+sp.cos(2.0*y))*sp.exp(-4.0*t)
    p_f = sp.lambdify([x, y, t], p, 'numpy')       
    
    ux = sp.diff(u, x)
    ux_f = sp.lambdify([x, y, t], ux, 'numpy')        
    uy = sp.diff(u, y)
    uy_f = sp.lambdify([x, y, t], uy, 'numpy')       
    ut = sp.diff(u, t)
    ut_f = sp.lambdify([x, y, t], ut, 'numpy')    
    vx = sp.diff(v, x)
    vx_f = sp.lambdify([x, y, t], vx, 'numpy')    
    vy = sp.diff(v, y)
    vy_f = sp.lambdify([x, y, t], vy, 'numpy')    
    vt = sp.diff(v, t)
    vt_f = sp.lambdify([x, y, t], vt, 'numpy')    
    px = sp.diff(p, x)
    px_f = sp.lambdify([x, y, t], px, 'numpy')    
    py = sp.diff(p, y)
    py_f = sp.lambdify([x, y, t], py, 'numpy')    
    pt = sp.diff(p, t)
    pt_f = sp.lambdify([x, y, t], pt, 'numpy')   
    
    uxx = sp.diff(ux, x)
    uxx_f = sp.lambdify([x, y, t], uxx, 'numpy')      
    uyy = sp.diff(uy, y)
    uyy_f = sp.lambdify([x, y, t], uyy, 'numpy')      
    vxx = sp.diff(vx, x)
    vxx_f = sp.lambdify([x, y, t], vxx, 'numpy')      
    vyy = sp.diff(vy, y)    
    vyy_f = sp.lambdify([x, y, t], vyy, 'numpy')  
    
    u_val = u_f(xvec, yvec, tvec)
    v_val = v_f(xvec, yvec, tvec) 
    p_val = p_f(xvec, yvec, tvec)
    
    ux_val = ux_f(xvec, yvec, tvec)
    uy_val = uy_f(xvec, yvec, tvec)
    ut_val = ut_f(xvec, yvec, tvec)
    vx_val = vx_f(xvec, yvec, tvec) 
    vy_val = vy_f(xvec, yvec, tvec)
    vt_val = vt_f(xvec, yvec, tvec)
    px_val = px_f(xvec, yvec, tvec)
    py_val = py_f(xvec, yvec, tvec)  
    pt_val = pt_f(xvec, yvec, tvec)  
    
    uxx_val = uxx_f(xvec, yvec, tvec)
    uyy_val = uyy_f(xvec, yvec, tvec)
    vxx_val = vxx_f(xvec, yvec, tvec)
    vyy_val = vyy_f(xvec, yvec, tvec)     
    
    return {'u':u_val, 'v':v_val, 'p':p_val,
            'ux':ux_val, 'uy':uy_val, 'ut':ut_val,
            'vx':vx_val, 'vy':vy_val, 'vt':vt_val,
            'px':px_val, 'py':py_val, 'pt':pt_val,
            'uxx':uxx_val, 'uyy':uyy_val,
            'vxx':vxx_val, 'vyy':vyy_val}





"""
for testing
"""
if __name__ == "__main__":

    ###
    a  = 1
    
    
    
    