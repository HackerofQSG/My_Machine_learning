# -*- coding: utf-8 -*-
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

in_size = 2
out_size = 1
hidden_size = 5

steps = 20

def add_layer(inputs, in_size, out_size):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
        return wx_plus_b, weights, biases

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, shape = (None, in_size), name = 'x_input')
    ys = tf.placeholder(tf.float32, shape = (None, out_size), name = 'y_input')

layer_1, w_1, b_1 = add_layer(xs, in_size, hidden_size)
prediction, w_2, b_2 = add_layer(layer_1, hidden_size, out_size)

with tf.name_scope('loss'):
    loss = - tf.reduce_mean(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 <1)] for (x1, x2) in X]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('w_1 = ', end = '')
    print(sess.run(w_1))
    print('b_1 = ', end = '')
    print(sess.run(b_1))
    print('w_2 = ', end = '')
    print(sess.run(w_2))
    print('b_2 = ', end = '')
    print(sess.run(b_2))

    for i in range(steps):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict = {xs: X[start:end], ys: Y[start:end]})
    print('Optimization is finished!')
    print('w_1 = ', end = '')
    print(sess.run(w_1))
    print('b_1 = ', end = '')
    print(sess.run(b_1))
    print('w_2 = ', end = '')
    print(sess.run(w_2))
    print('b_2 = ', end = '')
    print(sess.run(b_2))


