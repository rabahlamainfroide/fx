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

def _add_conv_layers(inputs,params, mode):
 convolved = tf.reshape(inputs, [-1, 36, 1])
 for i in range(len(params.num_conv)):
   convolved_input = convolved
   if params.batch_norm:
     convolved_input = tf.layers.batch_normalization(
         convolved_input,
         training=(mode == tf.estimator.ModeKeys.TRAIN))
   # Add dropout layer if enabled and not first convolution layer.
   if i > 0 and params.dropout:
     convolved_input = tf.layers.dropout(
         convolved_input,
         rate=params.dropout,
         training=(mode == tf.estimator.ModeKeys.TRAIN))
   convolved = tf.layers.conv1d(
       convolved_input,
       filters=params.num_conv[i],
       kernel_size=params.conv_len[i],
       activation=tf.nn.relu,
       strides=1,
       padding="same",
       name="conv1d_%d" % i)
   return convolved

def _add_regular_rnn_layers(convolved, params):
  """Adds RNN layers."""
  if params.cell_type == "lstm":
    cell = tf.nn.rnn_cell.BasicLSTMCell
  cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
  cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
  if params.dropout > 0.0:
    cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
    cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
  outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
      cells_fw=cells_fw,
      cells_bw=cells_bw,
      inputs=convolved,
      dtype=tf.float32,
      scope="rnn_classification")
  return outputs

def _add_rnn_layers(convolved, params):
  outputs = _add_regular_rnn_layers(convolved, params)
  outputs = tf.reduce_sum(outputs, axis=1)
  return outputs

def _add_fc_layers(final_state, params):
    return tf.layers.dense(final_state, params.num_classes)

def model(inputs, params, mode):
    convolved = _add_conv_layers(inputs=inputs, mode=mode, params=params)
    final_state = _add_rnn_layers(convolved=convolved, params=params)
    logits = _add_fc_layers(final_state=final_state, params=params)
    return logits
