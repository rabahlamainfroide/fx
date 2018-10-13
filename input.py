
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

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
from sklearn import preprocessing

def parse_tfexample_fn(example_proto, mode, p_wind_size):
    feature_to_type = {
        'p_wind' : tf.FixedLenFeature([p_wind_size * 4], dtype=tf.float32),
        'day_week' : tf.FixedLenFeature([1], dtype=tf.int64),
        'day_month' : tf.FixedLenFeature([1], dtype=tf.int64),
        'hour' :tf.FixedLenFeature([1], dtype=tf.int64)
        }

    if mode != tf.estimator.ModeKeys.PREDICT:
      feature_to_type["label"] = tf.FixedLenFeature([1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = parsed_features["label"]
    return parsed_features, labels


def get_input_fn(mode, tfrecord_pattern,  p_wind_size,batch_size, num_epochs):
  def input_fn():

        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern +'*')
        dataset = tf.data.TFRecordDataset(dataset)
        dataset = dataset.take(10000)
        dataset = dataset.prefetch(10000)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(functools.partial(parse_tfexample_fn, mode=mode, p_wind_size=p_wind_size),num_parallel_calls=12)
        dataset = dataset.batch(batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

  return input_fn

#
# input_fn= get_input_fn(tf.estimator.ModeKeys.TRAIN, "tfrecord/train/EURUSD_M5_36_48_15_25_0", 36, 1,1)
# features, labels= input_fn()
#
# features = features['p_wind']
#
# features = tf.reshape(features, [-1, 4, 36])
#
# features = features[: , 3 ,: ]
# features = tf.reshape(features, [-1,  36,1])
#
#
#
# with tf.Session() as sess :
#         print(sess.run(features))
# import os
# print(os.getcwd())
