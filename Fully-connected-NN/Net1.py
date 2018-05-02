
#全连接神经网络对于手写字体的识别

import pickle
import gzip
import random
import numpy as np

#下面函数是对于数据集的导入

def load_data():
    file = 'D:\python_machine-learning\DataSets\Mnist/mnist.pkl.gz'
    f = gzip.open(file,'rb')
    traning_data,validation_data,test_data = pickle.load(f,encoding='iso-8859-1')
    f.close()
    return (traning_data,validation_data,test_data)

def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs,training_results))
    validaton_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validaton_data = list(zip(validaton_inputs,va_d[1]))
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs,te_d[1]))
    return (training_data,validaton_data,test_data)

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e


class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)#代表有几层神经元
        self.sizes = sizes#sizes代表各层神经元的数量
        self.biaes = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes[:-1],sizes[1:])]#初始化权重


    def feedforward(self,a):#进行全连接神经网络的前向传播
        for b,w in zip(self.biaes,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

#下面是随即梯度下降算法，首先，training_data 是⼀个(x, y) 元组的列表，
# 表⽰训练输⼊和其对应的期望输出。变量epochs 和
# mini_batch_size 是 迭代期数量，和采样时的⼩批量数据的⼤⼩。eta 是学习速率，

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)#训练数据的大小
        for j in range(epochs):#迭代次数
            random.shuffle(training_data)#将训练数据集的顺序打乱
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]#将数据集随机采样，大小为mini_batch_size，放在mini_batches中
            for mini_batch in mini_batches:#对于每一个小的训练数据集，进行一次梯度下降
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print('Epoch:{}:{}/{}'.format(j,self.evaluate(test_data),n_test))
            else:
                print('Epoch {} complete'.format(j))


    def update_mini_batch(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch)*nw)
                        for w,nw in zip(self.weights,nabla_w)]
        self.biaes = [b-(eta/len(mini_batch)*nb)
                         for b,nb in zip(self.biaes,nabla_b)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs =[]
        for b,w in zip(self.biaes,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)#下一层的输入值
            activations.append(activation)#activation[-1]是最后的输出值
        delta = self.cost_derivative(activation[-1],y)*\
            sigmoid_prime(zs[-1])#求导
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)


    def evaluate(self,test_data):#评估函数
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y)\
                       in test_data]
        return sum(int(x ==y) for (x,y) in test_results)


    def cost_derivative(self,output_activations,y):#计算偏差的函数（输出值-真实值）
        return (output_activations-y)


def sigmoid(z):#激活函数
    return 1.0 / (1.0+np.exp(-z))


def sigmoid_prime(z):#对于激活函数（sigmoid函数）的求导
    return sigmoid(z) * (1-sigmoid(z))


# training_data, validation_data, test_data = load_data_wrapper()
# net = Network([784,50,10])
# net.SGD(training_data=training_data,epochs=30,mini_batch_size=10,eta=2.0,test_data=test_data)
