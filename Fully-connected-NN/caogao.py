import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from sklearn import svm
from  numpy import *

def sigmoid(z):#激活函数
    return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):#对于激活函数（sigmoid函数）的求导
    return sigmoid(z) * (1-sigmoid(z))


def cost_derivative(self,output_activations,y):#计算偏差的函数（输出值-真实值）
    return (output_activations-y)


W1 = [[0.15,0.2],
      [0.25,0.3]]


W2 = [[0.4,.45],
      [0.5,0.55]]
biaes = array([0.35,0.6])
weights = array([W1,W2])

y = [0.01,0.99]
nabla_b = [np.zeros(b.shape) for b in biaes]
nabla_w = [np.zeros(w.shape) for w in weights]


activation = [0.05,0.1]
activations = [activation]
zs = []
for b, w in zip(biaes, weights):
    z = np.dot(w, activation) + b
    print(z)
    zs.append(z)
    activation = sigmoid(z)  # 下一层的输入值
    activations.append(activation)  # activation[-1]是最后的输出值

delta = cost_derivative(activation[-1], y) * \
        sigmoid_prime(zs[-1])  # 求出的是输出误差（BP1）
nabla_b[-1] = delta
nabla_w[-1] = np.dot(delta, activations[-2].transpose())

