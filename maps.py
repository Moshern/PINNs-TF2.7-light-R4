# -*- coding: utf-8 -*-
"""
Created on Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.layers import Layer, Activation, Add, Lambda, multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.activations import sigmoid, tanh, relu
import tensorflow as tf


def swish(x, beta=0.1):
    """
    activation swish
    """
    return x * K.sigmoid(10 * beta * x)


def myswish(x):
    """
    activation swish
    """
    return x * K.sigmoid(x)



class Swish(Layer):
    """
    define the swish layer, beta is a trainable variable
    """
    def __init__(self, beta=0.1, trainable=False, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.beta_factor = self.add_weight(name='beta_factor', 
                                      shape=(1, 1),
                                      initializer= initializers.Constant(self.beta),
                                      trainable=self.trainable)

        super(Swish, self).build(input_shape)

    def call(self, inputs, mask=None):
        return swish(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
    
    


def batchnorm_activation(x, training_func=None):
    """
    return batchnorm and  activation
    batch normalize the data before activation
    """
    if training_func is None:
        # y = BatchNormalization()(x)
        # y = Activation(swish)(x)
        y = Swish(beta=0.1,trainable=True)(x)
        # y = Activation(K.tanh)(x)
        # y = Activation(K.sigmoid)(x)
        # y = LeakyReLU()(x)
        # y = ReLU()(x)
        # y = tanh(x)      
    else:     
        x0 = multiply([x,training_func])
        y = multiply([x, K.sigmoid(x0)])
    
    return y







def res_block(input_tensor, input_units, scale=0.2, training_func=None):
    x = Dense(units=input_units, activation=None, 
              kernel_regularizer=None)(input_tensor)
    x = batchnorm_activation(x, training_func=training_func)

    x = Dense(units=input_units, activation=None,
              kernel_regularizer=None)(x)
    
    if scale:
        """
        lambda匿名函数的格式：冒号前是参数，可以有多个，用逗号隔开，冒号右边的为表达式。
        其实lambda返回值是一个函数的地址，也就是函数对象。
        
        Lambda:仅仅对数据进行变换，不学习
        x = x*scale
        """
        x = Lambda(lambda t: t * scale)(x)  
    # equivalent to `added = tf.keras.layers.add([x1, x2])`
    x = Add()([x, input_tensor])
    x = batchnorm_activation(x, training_func=training_func)

    return x




def generator(layers, norm_paras, maptype='rnn'):
    # regularizer
    # gamma = 0.01
    # input data
    # # Now the model will take as input arrays of shape (None, layers[0])
    inputs = Input(shape=(layers[0],))
    # normalized to [-1 1]
    lb = norm_paras[0, 0:layers[0]]
    ub = norm_paras[1, 0:layers[0]]
    input_scale = 2.0*(inputs - lb)/(ub - lb) - 1.0
    
    
    # 训练激活函数中的自适应系数
    # 构建全链接网络，input_scale[1:-1,:]只和空间坐标有关
    # act_layer = 3*[layers[1]]
    # act_y = input_scale[:,1:4]
    # for width in act_layer[0:-1]:
    #     act_y = Dense(units=width, activation=None,
    #               kernel_regularizer=None)(act_y)
    #     act_y = myswish(act_y)
    # width = act_layer[-1]    
    # act_y = Dense(units=width, activation=None,
    #           kernel_regularizer=None)(act_y) 
    
    act_y = None
         
    if maptype.lower() == 'fnn':
        # Densely Connected Networks 
        x = input_scale
        for width in layers[1:-1]:
            x = Dense(units=width, activation=None,
                      kernel_regularizer=None)(x)
            #x = batchnorm_activation(x)
            x0 = multiply([x,act_y])
            x = multiply([x, K.sigmoid(x0)])
            #x0 = sigmoid(10.0*x0)
            #x = tanh(x)
            #x = Multiply([x, x0])
        width = layers[-1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(x) 
        # return the model
        return Model(inputs=inputs, outputs=x)
    
    
    elif maptype.lower() == 'rnn':
        num_layers = len(layers)
        block_size = int((num_layers-1-2)/2)
                
        # Residual Densely Connected Networks  
        width = layers[1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(input_scale)
        x = batchnorm_activation(x, training_func=act_y)
        
        for b in range(0,block_size):
            width = layers[2*b+2] 
            x = res_block(x, width, scale=1.0, training_func=act_y)
        width = layers[-1]     
        x = Dense(units=width, activation=None,
                  kernel_regularizer=None)(x)  
        # return the model
        return Model(inputs=inputs, outputs=x)        