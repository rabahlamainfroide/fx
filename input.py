

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



def parse_tfexample_fn(example_proto, mode, p_wind_size, f_wind_size ):
    """Parse a single record which is expected to be a tensorflow.Example."""
    feature_to_type = {
        'p_wind' : tf.FixedLenFeature([p_wind_size * 4], dtype=tf.float32),
        'f_wind' : tf.FixedLenFeature([f_wind_size * 4], dtype=tf.float32),
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


def get_input_fn(mode, tfrecord_pattern,  p_wind_size,f_wind_size,batch_size, num_epochs):
  def input_fn():
          with open("tfrecord/" + 'log.csv', 'r') as log_file:
              line = log_file.readline()
              line = np.array(line.split(','))
              min_of_classes = np.amin(np.ndarray.astype(line[0:3], int))

          dataset_0 = tf.data.TFRecordDataset.list_files(tfrecord_pattern +'0*')
          dataset_1= tf.data.TFRecordDataset.list_files(tfrecord_pattern + '1*')
          dataset_2= tf.data.TFRecordDataset.list_files(tfrecord_pattern + '2*')

          dataset_0 = tf.data.TFRecordDataset(dataset_0)
          dataset_1 = tf.data.TFRecordDataset(dataset_1)
          dataset_2 = tf.data.TFRecordDataset(dataset_2)

          dataset_0 = dataset_0.take(min_of_classes)
          dataset_1 = dataset_1.take(min_of_classes)
          dataset_2 = dataset_2.take(min_of_classes)

          dataset = dataset_2.concatenate(dataset_0).concatenate(dataset_1)

          if mode == tf.estimator.ModeKeys.TRAIN:
           dataset = dataset.shuffle(buffer_size= 30000000)
          dataset = dataset.repeat(num_epochs)
          dataset = dataset.map(
              functools.partial(parse_tfexample_fn, mode=mode, p_wind_size=p_wind_size, f_wind_size=f_wind_size),
              num_parallel_calls=12)
          dataset = dataset.prefetch(10000)
          dataset = dataset.batch(batch_size)
          features, labels = dataset.make_one_shot_iterator().get_next()
          return features['p_wind'] , labels
  return input_fn


  



input_fn= get_input_fn(tf.estimator.ModeKeys.TRAIN, "tfrecord/EURUSD_M5_36_48_15_25_*", 36, 48, 100,2)
p_wind, label =input_fn()



# with tf.Session() as sess :
#
#     p_wind =  tf.reshape(p_wind, [-1, 4, 36])
#     label = tf.reshape(label, [-1])
#     label = sess.run(label)
#
#     label=np.array(label)
#
#     print(collections.Counter(label))
