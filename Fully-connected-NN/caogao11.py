import tensorflow as tf
import numpy as np

# import input_data
# mnist = input_data.read_data_sets("D:\python_machine-learning\DataSets\MNIST_data",one_hot=True)
#
# sess = tf.InteractiveSession()
#
# x = tf.placeholder("float",shape=[None,784])
# y_ = tf.placeholder("float",shape=[None,10])
#
# #hiden layer
# W1 = tf.Variable(tf.zeros([784,30]))
# b1 = tf.Variable(tf.zeros([30]))
# Hiden1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
#
# #out layer
# W2 = tf.Variable(tf.zeros([30,10]))
# b2 = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(Hiden1,W2) + b2)
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#
# sess.run(tf.initialize_all_variables())
#
# for i in range(1000):
#     batch = mnist.train.next_batch(50)
#     train_step.run(feed_dict={x:batch[0],y_:batch[1]})
#
#
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
# print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
for i in range(1,2):
    print("1122")