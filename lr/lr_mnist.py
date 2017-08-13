#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 zhaokai.huang <huangzhaokai@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

"""
Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/

"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import argparse
import sys


FLAGS = None
batch_size = 128
n_epochs = 30


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    writer = tf.summary.FileWriter('./my_graph/03/logistic_mnist', sess.graph)

    tf.global_variables_initializer().run()
    start_time = time.time()
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):  # train the model n_epochs times
        total_loss = 0
        # Train
        for _ in range(n_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
            total_loss += loss_batch
        print 'Average loss epoch {0}: {1}'.format(i, total_loss / n_batches)

    print 'Total time: {0} seconds'.format(time.time() - start_time)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
