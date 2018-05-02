#使用tensorflow来实现去噪自编码器
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#使用Xaiver initialization来进行权重的初始化,使其满足均值为0，方差为2/（Nin+Nout）
def xavier_init(fan_in,fan_out,constant=1):#fan_in位输入节点的数量,fan_out为输出节点的数量
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)
