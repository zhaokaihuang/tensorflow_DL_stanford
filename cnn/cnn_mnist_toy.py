#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 zhaokai.huang <huangzhaokai@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

"""
Starter code for CNN model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

https://www.tensorflow.org/versions/master/get_started/mnist/pros
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import argparse
import sys


FLAGS = None
model_path = "./cnn_mnist_model.ckpt"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # input: [batch, in_height, in_width, in_channels]
    # filter:[filter_height, filter_width, in_channels, out_channels]
    # strides: 对于图片，因为只有两维，通常strides取[1，stride，stride，1]
    # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # Args:
    #
    # value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the
    # input tensor.
    # padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
    # data_format: A string. 'NHWC' and 'NCHW' are supported.
    # name: Optional name for the operation.
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    # output volume size: (W - F + 2P) / S + 1
    # before depth slice, num of neurons: feature map (width * height) * depth
    # after depth slice(parameter sharing), num of neurons: out_channels - 32

    # num of weights: 5*5*1 weights + 1 bias
    # filter:[filter_height, filter_width, in_channels, out_channels]
    W_con1 = weight_variable([5, 5, 1, 32])
    b_con1 = bias_variable([32])
    h_con1 = tf.nn.relu(conv2d(x_image, W_con1) + b_con1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_con1)

    # second convolutional layer
    W_con2 = weight_variable([5, 5, 32, 64])
    b_con2 = weight_variable([64])
    h_con2 = tf.nn.relu(conv2d(h_pool1, W_con2) + b_con2)

    h_pool2 = max_pool_2x2(h_con2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    # a fully - connected layer with 1024 neurons from 7*7*64 neurons
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # To reduce overfitting, we will apply dropout before the readout layer.
    # We create a placeholder for the probability that a neuron's output is kept during dropout.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Create the model
    # Build the graph for the deep net
    y_conv, keep_prob = cnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./my_graph/cnn_mnist', sess.graph)
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        # 调用上一次保存过的参数
        # saver.restore(sess, model_path)

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print "Model saved in file: %s" % save_path

    print 'Total time: {0} seconds'.format(time.time() - start_time)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

