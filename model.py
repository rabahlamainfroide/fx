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

wind_size = 20
f_wind_size = 10

def _add_conv_layers(inputs, is_training, params):
    inputs = tf.reshape(inputs, [-1,wind_size,4])
    convolved = inputs
    for i in range(len(params.num_conv)):
      convolved_input = convolved
      if params.batch_norm:
        convolved_input = tf.layers.batch_normalization(
            convolved_input,
            training=is_training)
      # Add dropout layer if enabled and not first convolution layer.
      if i > 0 and params.dropout:
        convolved_input = tf.layers.dropout(
            convolved_input,
            rate=params.dropout,
            training=is_training)
      convolved = tf.layers.conv1d(
          convolved_input,
          filters=params.num_conv[i],
          kernel_size=params.conv_len[i],
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_%d" % i)
    return convolved

def _add_regular_rnn_layers(inputs, params):
    """Adds RNN layers."""
    if params.cell_type == "lstm":
      cell = tf.nn.rnn_cell.BasicLSTMCell
    elif params.cell_type == "block_lstm":
      cell = tf.contrib.rnn.LSTMBlockCell
    cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
    if params.dropout > 0.0:
      cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
      cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=inputs,
        dtype=tf.float32,
        scope="rnn_classification")
    return outputs
def _add_cudnn_rnn_layers(inputs, is_training, params):
    """Adds CUDNN LSTM layers."""
    # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
    convolved = tf.transpose(inputs, [1, 0, 2])
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=params.num_layers,
        num_units=params.num_nodes,
        dropout=params.dropout if is_training else 0.0,
        direction="bidirectional")
    outputs, _ = lstm(convolved)
    # Convert back from time-major outputs to batch-major outputs.
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs
def _add_rnn_layers(inputs, is_training, params):
    """Adds recurrent neural network layers depending on the cell type."""
    if params.cell_type != "cudnn_lstm":
      outputs = _add_regular_rnn_layers(inputs, params)
    else:
      outputs = _add_cudnn_rnn_layers(inputs, is_training, params)
    return outputs
def _add_fc_layers(inputs, num_classes):
    inputs = tf.reshape(inputs, [-1, wind_size * 128*2])

    inputs = tf.layers.dense(inputs, num_classes)
    return inputs



def model(inputs, num_classes, is_training, params):
      inputs = _add_conv_layers(inputs, is_training, params)


      inputs = _add_rnn_layers(inputs, is_training, params)
      inputs = _add_fc_layers(inputs, num_classes)
      return inputs
