from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from dateutil.parser import parse
import argparse
import sys
import functools

def _add_conv_layers(inputs, is_training):
    inputs = tf.layers.conv2d(inputs, filters= 256 , kernel_size=[4, 4], strides=1, padding="same",activation=tf.tanh, name= 'conv1')
    inputs = tf.layers.batch_normalization(inputs,training=is_training)
    inputs = tf.layers.conv2d(inputs, filters= 512, kernel_size=[4, 4], strides=1, padding="same",activation=tf.tanh, name= 'conv2')
    inputs = tf.layers.batch_normalization(inputs,training=is_training)
    inputs = tf.layers.average_pooling2d( inputs=inputs, pool_size=2, strides=2, padding='same')
    inputs = tf.layers.conv2d(inputs, filters= 1024 , kernel_size=[4, 4], strides=1, padding="same", activation=tf.tanh, name= 'conv3')
    inputs = tf.layers.conv2d(inputs, filters= 1024 , kernel_size=[4, 4], strides=1, padding="same",activation=tf.tanh, name= 'conv4')
    inputs = tf.layers.conv2d(inputs, filters= 1024 , kernel_size=[4, 4], strides=1, padding="same",activation=tf.tanh, name= 'conv5')
    inputs = tf.layers.average_pooling2d(inputs=inputs, pool_size=2, strides=2, padding='same')
    inputs = tf.layers.batch_normalization(inputs,training=is_training)
    return inputs

def _add_regular_rnn_layers(inputs, params, num_nodes=2000,  num_layers=5 ):
    inputs = tf.reshape(inputs, [-1,  params.p_wind_size,1])
    cell = tf.nn.rnn_cell.BasicLSTMCell
    cells_fw = [cell(num_nodes) for _ in range(num_layers)]
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    outputs, _ = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs, initial_state=None, scope="rnn_classification", dtype=tf.float32)
    output = tf.layers.dense(outputs[:, -1, :],  params.num_classes)
    return output


def _add_fc_layers(inputs, params):
    features = tf.reshape(features, [-1,params.p_wind_size])
    inputs = tf.layers.dense(inputs, 2048)
    inputs = tf.layers.dense(inputs, params.num_classes)
    return inputs

def model(inputs, is_training, params):
      #inputs = _add_conv_layers(inputs, is_training)
      inputs = _add_regular_rnn_layers(inputs, params)
      #inputs = _add_fc_layers(inputs,params)
      return inputs
