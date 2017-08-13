#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 zhaokai.huang <huangzhaokai@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn

# tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

import tensorflow as tf

a = tf.constant([2, 3], shape=[1, 2], name='a')
b = tf.constant([[1, 0], [2, 2]], name='b')
x = tf.add(a, b, name='add')
y = tf.matmul(a, b, name='matmul')

with tf.Session() as sess:
    x, y = sess.run([x, y])
    print x, y


# 1. The easiest way is initializing all variables at once:
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# 2. Initialize only a subset of variables:
# init_ab = tf.variables_initializer([a, b], name="init_ab")
# with tf.Session() as sess:
#     sess.run(init_ab)

# 3. Initialize a single variable
W = tf.Variable(tf.zeros([784, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)


# tf.placeholder(dtype, shape=None, name=None)
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([4, 5, 6], tf.float32)
c = tf.add(a, b)
with tf.Session() as sess:
    print sess.run(c, {a: [1, 2, 3]})
