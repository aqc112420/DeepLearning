import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = 2 * np.power(x_data,3)  + np.power(x_data,2) + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#Hiden layer1
Weight1 = tf.Variable(tf.random_normal([1,5]))
biases1 = tf.Variable(tf.zeros([1,5]) + 0.1)
Wx_plus_b1 = tf.matmul(xs,Weight1) + biases1
l1 = tf.nn.relu(Wx_plus_b1)

#Hiden layer2
Weight2 = tf.Variable(tf.random_normal([5,10]))
biases2 = tf.Variable(tf.zeros([1,10]) + 0.1)
Wx_plus_b2 = tf.matmul(l1,Weight2) + biases2
l2 = tf.nn.relu(Wx_plus_b2)

#out layer
Weight3 = tf.Variable(tf.random_normal([10,1]))
biases3 = tf.Variable(tf.zeros([1,1]) + 0.1)
prediction = tf.matmul(l2,Weight3) + biases3


#loss 表达式这里采用的是均方差而不用交叉熵

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

#优化策略
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_data,y_data)
    plt.ion()
    plt.show()

    for i in range(10000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 ==0:
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(1)
            print(i / 50 ,end=" ")
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))



