import tensorflow as tf
import numpy as np
import load_data
import math
sess=tf.Session()

def add_layer(inputs,in_size,out_size,activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

train_x,ImgNum1=load_data.load_image1()
train_y,ImgNum2=load_data.load_image2()

x_data=train_x.reshape((40000, 784))
y_data=train_y.reshape((40000, 784))


batch=1000
NumBatch=math.floor(ImgNum2/batch)

batch_xs = x_data[1:batch+1 ][:]
batch_ys = y_data[1:batch+1 ][:]

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,784])
keep_prob=tf.placeholder(tf.float32)

h1=add_layer(xs,784,784,activation_function=tf.nn.relu)
hidden=tf.nn.dropout(h1,keep_prob)
prediction=add_layer(hidden,784,784,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(batch_ys-prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in range(1,1000):
    for batch_id in range(1, 2):#NumBatch-1
        batch_xs = x_data[(batch_id - 1) * batch + 1:batch_id * batch + 1][:]
        batch_ys = y_data[(batch_id - 1) * batch + 1:batch_id * batch + 1][:]

        sess.run([train_step],feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.75})
 #       if i% 50 ==0:
        print(sess.run(loss,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.75}))
test_data = x_data[0:40000,:]
test_img = sess.run(prediction,feed_dict={xs:test_data,keep_prob:0.75})
test_img = test_img.T
y_data = y_data.T
print(test_img.shape)
print(y_data.shape)

# test
m = 40000
n = 784
threshold = float(0)
accuracy = float(0)
f1 = float(0)
tpr = float(0)
Max1 = float(0)
Max2 = float(0)
Max3 = float(0)
for threshold in range(0, 100, 1):
    thr = threshold / 100
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for j in range(1, m, 1):
        imgTest = y_data[:][j]
        imgGen = test_img[:][j]
        for k in range(1, n, 1):
            if imgGen[k] > thr:
                imgGen[k] = 1
                if imgGen[k] == imgTest[k]:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                imgGen[k] = 0
                if imgGen[k] == imgTest[k]:
                    TN = TN + 1
                else:
                    FN = FN + 1
    accuracy = (TN + TP) / (m * n)
    f1 = 2 * TP / (2 * TP + FN + FP)
    tpr = TP / (TP + FN)
    if accuracy > Max1:
        optAccThreshold = thr
        Max1 = accuracy
    if f1 > Max2:
        optF1Threshold = thr
        Max2 = f1
    if tpr > Max3:
        optTprThreshold = thr
        Max3 = tpr
print(f1.shape)
print(f1)
