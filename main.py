

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

from model import model
import input
import glob
from os.path import normpath


class SaverListener(tf.train.CheckpointSaverListener):
  DST_DIR = 'drive/python_project/forex_project_v7/model_dir/'
  LOG_DIR = ''

  def after_save(self, session, global_step_value):
      get_ipython().system_raw(
                'cp -r {} {} &'.format(self.LOG_DIR, self.DST_DIR))


def model_fn(features, labels, mode, params):
  features = features['p_wind']
  features = tf.reshape(features, [-1, 4, params.p_wind_size])
  features = features[: , 3 , :]

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN, params)
  predictions = {'classes': tf.argmax(logits, axis=1)}
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  labels= tf.reshape(labels,[-1])
  accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  metrics = {'accuracy': accuracy}

  recall= tf.metrics.recall(labels, predictions['classes'])
  tf.identity(recall[1], name='recall')
  tf.summary.scalar('recall', recall[1])


  precision= tf.metrics.precision(labels, predictions['classes'])
  tf.identity(precision[1], name='precision')
  tf.summary.scalar('precis', precision[1])

  accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  conf_matrix= tf.confusion_matrix(labels, predictions['classes'],3)
  conf_matrix= tf.reshape(conf_matrix, [-1,3,3,1])
  tf.identity(conf_matrix, name='confusion_matrix')
  tf.summary.image('confusion_matrix', tf.cast(conf_matrix,dtype=tf.float32))

  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
  train_op = tf.contrib.layers.optimize_loss( loss=cross_entropy,global_step= tf.train.get_global_step(),
                                                learning_rate=params.learning_rate, optimizer="Adam",
                                                clip_gradients=params.gradient_clipping_norm,
                                                summaries=["loss"])
  return tf.estimator.EstimatorSpec(mode=mode,
                                    predictions={"logits": logits, "predictions": predictions['classes']},
                                    loss=cross_entropy,
                                    train_op=train_op,
                                    eval_metric_ops=metrics)

def create_estimator_and_specs(run_config):

  """Creates an Experiment configuration based on the estimator and input fn."""
  model_params = tf.contrib.training.HParams(
      num_layers=FLAGS.num_layers,
      p_wind_size= FLAGS.pwind,
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

  # Saver hook
  if FLAGS.colab:
    saver = tf.train.Saver()
    listener = SaverListener()
    listener.LOG_DIR = FLAGS.model_dir
    saver_hook = tf.train.CheckpointSaverHook(FLAGS.model_dir+'1', listeners=[listener], saver=saver, save_steps=100)
    train_spec = tf.estimator.TrainSpec(input_fn=input.get_input_fn(mode=tf.estimator.ModeKeys.TRAIN, tfrecord_pattern='tfrecord/train/', batch_size=FLAGS.batch_size, p_wind_size=FLAGS.pwind,
                                                                    f_wind_size=FLAGS.fwind, num_epochs = 1000000), hooks=[saver_hook])
  else:
    train_spec = tf.estimator.TrainSpec(input_fn=input.get_input_fn(mode=tf.estimator.ModeKeys.TRAIN, tfrecord_pattern='tfrecord/train/', batch_size=FLAGS.batch_size, p_wind_size=FLAGS.pwind,
                                                                      f_wind_size=FLAGS.fwind, num_epochs = 1000000))

  eval_spec = tf.estimator.EvalSpec(input_fn=input.get_input_fn(mode=tf.estimator.ModeKeys.EVAL, tfrecord_pattern='tfrecord/eval/', batch_size=FLAGS.batch_size, p_wind_size=FLAGS.pwind, f_wind_size=FLAGS.fwind,
                                                                num_epochs = 1), start_delay_secs = 10000000)

  return estimator, train_spec, eval_spec


def main(unused_argv):

    output_name = FLAGS.pair +'_M'+ str(FLAGS.frame) +'_'+ str(FLAGS.pwind) +'_'+ str(FLAGS.fwind) +'_'+ str(FLAGS.loss) +'_'+ str(FLAGS.profit)
    train_path = os.path.join('tfrecord/train', output_name)
    eval_path = os.path.join('tfrecord/eval', output_name)

    if (glob.glob(train_path + '*') and glob.glob(eval_path + '*')) :
        print('print running estimator on tfrecord with patern :', train_path, 'and', eval_path )
        estimator, train_spec, eval_spec = create_estimator_and_specs(
            run_config=tf.estimator.RunConfig(
                model_dir=FLAGS.model_dir,
                save_checkpoints_secs=300,
                save_summary_steps=100))

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else :
        print(' no tfrecord file fitting this pattern : ', train_path, 'or', eval_path  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--pair",
      type=str,
      default="EURUSD",
      help="Pair")
  parser.add_argument(
      "--frame",
      type=int,
      default=5,
      help="Frame")
  parser.add_argument(
      "--pwind",
      type=int,
      default="20",
      help="PWIND Value")
  parser.add_argument(
      "--fwind",
      type=int,
      default=10,
      help="FWIND Value")
  parser.add_argument(
      "--loss",
      type=int,
      default=5,
      help="Loss in PIP")
  parser.add_argument(
      "--profit",
      type=int,
      default=5,
      help="Profit in PIP")
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
      default=1000,
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
      default=0.00001,
      help="Learning rate used for training.")
  parser.add_argument(
      "--gradient_clipping_norm",
      type=float,
      default=None,
      help="Gradient clipping norm used during training.")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--steps",
      type=int,
      default=100000000,
      help="Number of training steps.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=300,
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
  parser.add_argument(
      "--output_shards",
      type=int,
      default=10,
      help="Number of shards for the output.")

  parser.add_argument(
      "--csv_dir",
      type=str,
      default="csv_dir",
      help="Directory where the ndjson files are stored.")

  parser.add_argument(
      "--colab",
      type=bool,
      default=False,
      help="If running on colab.")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
