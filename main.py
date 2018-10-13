

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from dateutil.parser import parse
import argparse
import sys
import functools

import model
import input


tfrecord_patten=r"C:\Users\Chak\Desktop\python_project\forex_project\tfrecord\tfrecord*"


def model_fn(features, labels, mode, params):




  logits = model.model(features, 3, mode == tf.estimator.ModeKeys.TRAIN, params )
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  labels= tf.reshape(labels,[-1])
  cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))

  train_op = tf.contrib.layers.optimize_loss(
     loss=cross_entropy,
     global_step= tf.train.get_or_create_global_step(),
     learning_rate=params.learning_rate,
     optimizer="Adam",
     # some gradient clipping stabilizes training in the beginning.
     clip_gradients=params.gradient_clipping_norm,
     summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
  predictions = tf.argmax(logits, axis=1)
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={"logits": logits, "predictions": predictions},
      loss=cross_entropy,
      train_op=train_op,
      eval_metric_ops={"accuracy": tf.metrics.accuracy(tf.argmax(labels, axis=1) , predictions)})

def create_estimator_and_specs(run_config):

  """Creates an Experiment configuration based on the estimator and input fn."""
  model_params = tf.contrib.training.HParams(
      num_layers=FLAGS.num_layers,
      num_nodes=FLAGS.num_nodes,
      batch_size=FLAGS.batch_size,
      num_conv=FLAGS.num_conv,
      conv_len=FLAGS.conv_len,
      num_classes=3,
      learning_rate=FLAGS.learning_rate,
      gradient_clipping_norm=FLAGS.gradient_clipping_norm,
      cell_type=FLAGS.cell_type,
      batch_norm=FLAGS.batch_norm,
      dropout=FLAGS.dropout)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=model_params)

  train_spec = tf.estimator.TrainSpec(input_fn=input.get_input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      tfrecord_pattern=tfrecord_patten,
      batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

  eval_spec = tf.estimator.EvalSpec(input_fn=input.get_input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      tfrecord_pattern=tfrecord_patten,
      batch_size=FLAGS.batch_size))

  return estimator, train_spec, eval_spec



def main(unused_argv):


  estimator, train_spec, eval_spec = create_estimator_and_specs(
      run_config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_secs=300,
          save_summary_steps=100))
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--tfrecord_data",
      type=str,
      default="tfrecord/*",
      help="Path to training data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--eval_data",
      type=str,
      default="tfrecord/*",
      help="Path to evaluation data (tf.Example in TFRecord format)")

  parser.add_argument(
      "--num_layers",
      type=int,
      default=3,
      help="Number of recurrent neural network layers.")
  parser.add_argument(
      "--num_nodes",
      type=int,
      default=128,
      help="Number of node per recurrent network layer.")
  parser.add_argument(
      "--num_conv",
      type=str,
      default=[48, 64, 96],
      help="Number of conv layers along with number of filters per layer.")
  parser.add_argument(
      "--conv_len",
      type=str,
      default=[5, 5, 3],
      help="Length of the convolution filters.")
  parser.add_argument(
      "--cell_type",
      type=str,
      default="lstm",
      help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
  parser.add_argument(
      "--batch_norm",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.0001,
      help="Learning rate used for training.")
  parser.add_argument(
      "--gradient_clipping_norm",
      type=float,
      default=9.0,
      help="Gradient clipping norm used during training.")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--steps",
      type=int,
      default=100000,
      help="Number of training steps.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=120,
      help="Batch size to use for training/evaluation.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="model_dir",
      help="Path for storing the model checkpoints.")
  parser.add_argument(
      "--self_test",
      type="bool",
      default="False",
      help="Whether to enable batch normalization or not.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
