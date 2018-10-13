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



def _add_conv_layers(inputs, is_training,params):
    for i in range(len(params.num_conv)):
      convolved_input = inputs
      if params.batch_norm:
        convolved_input = tf.layers.batch_normalization(convolved_input,training=is_training)

      convolved = tf.layers.conv1d(
          convolved_input,
          filters=params.num_conv[i],
          kernel_size=params.conv_len[i],
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_%d" % i)
      convolved = tf.layers.average_pooling1d( inputs=convolved, pool_size=2, strides=1, padding='VALID')
    return convolved

def _add_regular_rnn_layers(inputs, params):
    """Adds RNN layers."""
    if params.cell_type == "lstm":
      cell = tf.nn.rnn_cell.BasicLSTMCell
    elif params.cell_type == "block_lstm":
      cell = tf.contrib.rnn.LSTMBlockCell
    cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    #if params.dropout > 0.0:
    #  cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    _, state = tf.nn.dynamic_rnn(stacked_rnn_cell, inputs,scope="rnn_classification", dtype=tf.float32)
    return state[params.num_layers-1].h


def _add_rnn_layers(inputs, is_training, params):
    """Adds recurrent neural network layers depending on the cell type."""
    if params.cell_type != "cudnn_lstm":
      outputs = _add_regular_rnn_layers(inputs, params)
    else:
      outputs = _add_cudnn_rnn_layers(inputs, is_training, params)
    return outputs
def _add_fc_layers(inputs, params):
    inputs = tf.reshape(inputs, [-1, params.num_nodes ])
    inputs = tf.layers.dense(inputs, 1024, activation=tf.nn.relu)
    inputs = tf.layers.dense(inputs, params.num_classes)
    return inputs



def model(inputs, is_training, params):
      #inputs = _add_conv_layers(inputs, is_training, params)
      inputs = _add_rnn_layers(inputs, is_training, params)
      inputs = _add_fc_layers(inputs,params )
      return inputs
