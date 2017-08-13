#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 zhaokai.huang <huangzhaokai@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn


import tensorflow as tf

a = tf.constant(2, name='a1')
b = tf.constant(3, name='b1')
x = tf.add(a, b, name='add')

with tf.Session() as sess:
    # add this line to use tensorboard
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(x)

writer.close()

# tensorboard --logdir = "./graphs"
# localhost:6060
