

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import time
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from dateutil.parser import parse
import argparse
import sys
import functools


buffer_size = 6
num_epochs = 1

def parse_tfexample_fn(example_proto, mode, p_wind_size ):
    """Parse a single record which is expected to be a tensorflow.Example."""
    feature_to_type = {
        'p_wind' : tf.FixedLenFeature([p_wind_size * 4], dtype=tf.float32),
        'hour' :tf.FixedLenFeature([1], dtype=tf.int64),
        'day_week' : tf.FixedLenFeature([1], dtype=tf.int64),
        'day_month' : tf.FixedLenFeature([1], dtype=tf.int64)}

    if mode != tf.estimator.ModeKeys.PREDICT:
      feature_to_type["label"] = tf.FixedLenFeature([1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    #features = parsed_features["p_wind"]
    labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = parsed_features["label"]
      #labels = tf.one_hot(labels, 3)
    return parsed_features, labels


def get_input_fn(mode, tfrecord_pattern, batch_size, p_wind_size):
  def input_fn():
          dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
          if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=3)
          dataset = dataset.repeat()
          # Preprocesses 10 files concurrently and interleaves records from each file.
          dataset = dataset.interleave(
             tf.data.TFRecordDataset,
              cycle_length=10,
              block_length=1)
          dataset = dataset.map(
              functools.partial(parse_tfexample_fn, mode=mode, p_wind_size=p_wind_size),
              num_parallel_calls=1)
          dataset = dataset.prefetch(1)
          if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1)
          # Our inputs are variable length, so pad them.
          dataset = dataset.batch(batch_size)
          features, labels = dataset.make_one_shot_iterator().get_next()

          return features['p_wind'], labels
  return input_fn
