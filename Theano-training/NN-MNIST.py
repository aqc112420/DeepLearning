#三层神经网络模型，用于MNIST数据集的识别

import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression,load_data
class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(low=np.sqrt(6./
                                                          n_in+n_out),
                                  high=np.sqrt(6./(n_in+n_out)),
                                  size=(n_in, n_out)),
                                  dtype=theano.config.floatX
                                  )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values,name="W",borrow=True)

        if b is None:
            b_values = np.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_values,name="b",borrow=True)
        self.W = W
        self.b = b

        lin_output = T.dot(input,self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))

        self.params = [self.W,self.b]


class MLP(object):
    def __init__(self,rng,input,n_in,n_hidden,n_out):

        self.hiddenLayer = HiddenLayer(rng=rng,
                                       input=input,
                                       n_in=n_in,
                                       n_out=n_hidden,
                                       activation=T.tanh)


        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output,
                                                     n_in=n_hidden,n_out=n_out)
        self.L1 = (abs(self.hiddenLayer.W).sum()+abs(self.logRegressionLayer.W).sum())

        self.L2_sqr = ((self.hiddenLayer.W ** 2).sum()
                       + (self.logRegressionLayer.W ** 2).sum())
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input

file = "D:\python_machine-learning\DataSets\Mnist/mnist.pkl.gz"

def test_mlp(learning_rate=0.01,L1_reg=0.00,L2_reg=0.0001,n_epochs=1000,
             dataset=file,batch_size=20,n_hidden=500):
    datasets = load_data(dataset)
    train_set_x,train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng,input=x,n_in=28 * 28,n_hidden=n_hidden,n_out=10)

    cost = (classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr)

    gparams = [T.grad(cost,param) for param in classifier.params]
    updates = [
        (param,param - learning_rate * gparam)
        for param,gparam in zip(classifier.params,gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x:train_set_x[index *batch_size:(index + 1) * batch_size],
            y:train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    epoch = 0.0
    while (epoch < 10):
        cost = 0.0
        for minibatch_index in range(n_train_batches):
            cost += train_model(minibatch_index)
        print("epoch:",epoch,"  error:",cost/n_train_batches)
        epoch = epoch + 1


test_mlp()