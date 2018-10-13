import pandas as pd
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from dateutil.parser import parse
from datetime import timedelta
import os
import argparse
import sys
import glob
from random import shuffle
import threading



def _pick_output_shard(output_shards):
  if (output_shards == 1):
    return 0
  else :
    return np.random.randint(0, output_shards - 1)

def dict_to_train_feature(features):
    features["p_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=features["p_wind"].flatten()))
    features["f_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=features["f_wind"].flatten()))
    features["day_week"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features["day_week"]]))
    features["day_month"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features["day_month"]]))
    features["hour"]= tf.train.Feature(int64_list=tf.train.Int64List(value=[features["hour"]]))
    features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features["label"]]))
    return tf.train.Features(feature=features)

def print_recurssevely(str):
        print(str, end='')
        print('\r', end='')
        time.sleep(0.2)

def minutes(datetime):
    return datetime.minute

class time_series (object):
    def __init__(self, df,p_wind_size,f_wind_size, profit, loss):
        self.df = df
        self.p_wind_size=p_wind_size
        self.f_wind_size=f_wind_size
        self.offset=0
        self.window_discarded=0
        self.lapswindow=timedelta(minutes=p_wind_size+f_wind_size-1)
        self.profit = profit
        self.loss = loss

    def window(self, frame, tolerance):

        def check_window():
            for i in range (self.df.shape[0]-self.offset-self.p_wind_size-self.f_wind_size):
                first = self.df.iloc[self.offset,0]
                last = self.df.iloc[self.offset +self.p_wind_size+ self.f_wind_size-1,0]
                lapse=last-first
                if ((lapse/frame - timedelta(minutes=self.p_wind_size+ self.f_wind_size-1)) > timedelta(minutes=tolerance))  :
                    self.window_discarded +=1
                    self.offset += 1
                    if (self.offset +self.f_wind_size+self.p_wind_size>= self.df.shape[0]):
                        return False
                else:
                    return True
        if not check_window():
            return None,None, None, None, None, None

        a = label_generator(self.profit, self.loss)

        p_wind= np.array(self.df.iloc[self.offset:self.offset + self.p_wind_size,2:6])
        f_wind=np.array(self.df.iloc[self.offset + self.p_wind_size-1:self.offset +self.p_wind_size+ self.f_wind_size-1,2:6])
        p_wind = ((p_wind - p_wind[0,3]) * 10000).astype(int)
        f_wind = ((f_wind - f_wind[0,3]) * 10000).astype(int)
        time_date = self.df.iloc[self.offset + self.p_wind_size,0]
        vol = self.df.iloc[self.offset + self.p_wind_size,6]
        self.offset=self.offset+1
        if (self.offset +self.f_wind_size>= self.df.shape[0]):
            return False
        features =  {}

        features["p_wind"] = np.rot90(p_wind)
        features["f_wind"] = np.rot90(f_wind)
        features["day_week"] = time_date.weekday()
        features["day_month"] = time_date.day
        features["hour"]= time_date.hour
        features["label"] = a.get_label(features["f_wind"], self.f_wind_size)
        return features


    def reset_offset(self):
        self.offset = 0

class label_generator(object):
    def __init__(self,profit, loss):
        self.sell_counter = 0
        self.loss= loss
        self.profit=profit
        self.buy_counter = 0
        self.neutral_counter = 0

    def get_label(self,f_wind, f_wind_size):

        lowest_low=np.amin(f_wind[1])
        highst_high=np.amax(f_wind[3])
        close_price=f_wind[1,f_wind_size-1]
        open_price=f_wind[3,0]
        up_swing=highst_high - open_price
        down_swing=open_price - lowest_low
        if (up_swing >= self.profit) and (down_swing <=self.loss) :
            label = 1
            self.buy_counter+=1
        else :
            if (up_swing <= self.loss)  and (down_swing >=self.profit):
                label = 2
                self.sell_counter+=1
            else :
                label=0
                self.neutral_counter+=1
        return label


def feature_dict_to_tfrecord(feature_dict, writers, writers_index):
    features = {}
    features["p_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_dict["p_wind"].flatten()))
    features["f_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_dict["f_wind"].flatten()))
    features["day_week"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dict["day_week"]]))
    features["day_month"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dict["day_month"]]))
    features["hour"]= tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dict["hour"]]))
    features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dict["label"]]))
    f = tf.train.Features(feature=features)
    example = tf.train.Example(features=f)
    writers[writers_index].write(example.SerializeToString())

def csv_to_data_frame(csv_path, frame):
    df = pd.read_csv(csv_path, parse_dates=[[1,2]], infer_datetime_format= True , delimiter = ',')
    df =df.loc[df['<DTYYYYMMDD>_<TIME>'].apply(minutes) % frame==0]
    return df



def from_time_series_to_tfrecord(tm, writers, writers_index, thread_index):
    lost = np.array([0, 0, 0])
    stored = np.array([0, 0, 0])
    while True :
        min_of_classes = np.amin(np.array(get_min_of_classes(tm, thread_index)))
        tm.reset_offset()
        list_buffers, last, lost_iter, stored_iter = three_class_buffers(tm, min_of_classes, thread_index)
        three_class_buffers_to_tfrecord(list_buffers, writers, writers_index, last)
        lost += np.array(lost_iter)
        stored += np.array(stored_iter)
        if(last == True):
            print('thread ', thread_index )
            print('lost |', lost[0], '|', lost[1], '|', lost[2])
            print('stored |', stored[0], '|', stored[1], '|', stored[2])
            break

def get_min_of_classes(tm, thread_index):
    i = 0
    classes = [0, 0, 0]
    max_data=tm.df.shape[0]-tm.p_wind_size-tm.f_wind_size
    while True:
        features= tm.window(FLAGS.frame, FLAGS.tolerance)
        if (tm.df.shape[0] - tm.offset - tm.p_wind_size - tm.f_wind_size) <=0 :
            return classes
        if (i % 100000 == 0) :
            print('thread', thread_index, ' counting min_lasses : ' , int(tm.offset * 100/max_data), ' %')
        if features['label'] == 0:
            classes[features['label']] += 1
        elif features['label'] == 1:
            classes[features['label']] += 1
        elif features['label'] == 2:
            classes[features['label']] += 1
        i+=1


def three_class_buffers(tm, thresh, thread_index):
    lost = [0, 0, 0]
    stored = [0, 0, 0]
    buffer_0 = []
    buffer_1 = []
    buffer_2 = []
    max_data=tm.df.shape[0]-tm.p_wind_size-tm.f_wind_size
    i=0
    while True:
        features= tm.window(FLAGS.frame, FLAGS.tolerance)
        if (tm.df.shape[0] - tm.offset - tm.p_wind_size - tm.f_wind_size) <=0 :
            return [buffer_0, buffer_1, buffer_2], True, lost, stored
        if (i % 100000 == 0) :
            print('thread', thread_index, 'writing to tfrecord: ' , int(tm.offset * 100/max_data), ' %')
        if features['label'] == 0 and len(buffer_0) < thresh:
            stored[features['label']] += 1
            buffer_0.append(features)
        elif features['label'] == 1 and len(buffer_1) < thresh:
            stored[features['label']] += 1
            buffer_1.append(features)
        elif features['label'] == 2 and len(buffer_2) < thresh:
            stored[features['label']] += 1
            buffer_2.append(features)
        else:
            lost[features['label']] += 1
        if len(buffer_0) == len(buffer_1) == len(buffer_2) == 10:
            break
        i+=1
    return [buffer_0, buffer_1, buffer_2], False, lost, stored
    def writer_generator(output_path, output_shards):
        writers = []
        for i in range(output_shards):
          writers.append(tf.python_io.TFRecordWriter((output_path +'_' + str(i) )))
        return writers


def three_class_buffers_to_tfrecord(list_buffers, writers, writers_index, last):
    big_buffer=[]
    for i in range(len(list_buffers[0])):
        big_buffer.append(list_buffers[0][i])
        big_buffer.append(list_buffers[1][i])
        big_buffer.append(list_buffers[2][i])

    for features in big_buffer:
        feature_dict_to_tfrecord(features, writers, writers_index)


def main(argv):



    def create_dataset(train_fraction):
        csv_name = FLAGS.pair +'_M1.csv'
        csv_path = os.path.join(FLAGS.csv_dir, csv_name)
        df = csv_to_data_frame(csv_path, FLAGS.frame)
        df_train, df_eval = np.split(df, [int(train_fraction * len(df))])
        output_name = FLAGS.pair +'_M'+ str(FLAGS.frame) +'_'+ str(FLAGS.pwind) +'_'+ str(FLAGS.fwind) +'_'+ str(FLAGS.loss) +'_'+ str(FLAGS.profit)

        train_output_path = os.path.join(FLAGS.output_path, 'train', output_name)
        eval_output_path = os.path.join(FLAGS.output_path, 'eval', output_name)
        print('train_dataset : ' , int(len(df_train) * 100/len(df)), '%')
        print('test_dataset : ' , int(len(df_eval) * 100/len(df)), '%')
        writers = []

        for i in range(3):
              writers.append(tf.python_io.TFRecordWriter((train_output_path +'_' + str(i) )))
        writers.append(tf.python_io.TFRecordWriter((eval_output_path +'_' + str(i) )))
        df_list = [None, None, None, None]
        df_list[0], df_list[1], df_list[2] = np.split(df_train, [int(0.33 * len(df_train)), int(0.66 * len(df_train))])
        df_list[3] = df_eval
        ts_list = []
        for i in range(4):
            ts_list.append(time_series(df_list[i], FLAGS.pwind, FLAGS.fwind, FLAGS.profit, FLAGS.loss))
        threads = []
        for i in range(4):
            t = threading.Thread(target=from_time_series_to_tfrecord, args=(ts_list[i], writers, i, i ))
            threads.append(t)
            t.start()



    create_dataset(0.8)




if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "--write",
       type=bool,
       default=False,
       help="Write to TFRec")

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
       "--tolerance",
       type=int,
       default=3,
       help="Tolerance")
   parser.add_argument(
       "--pwind",
       type=int,
       default="20",
       help="PWIND Size")
   parser.add_argument(
       "--fwind",
       type=int,
       default=10,
       help="FWIND Size")
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
   parser.register("type", "bool", lambda v: v.lower() == "true")
   parser.add_argument(
      "--csv_dir",
      type=str,
      default="csv_dir",
      help="Directory where the ndjson files are stored.")
   parser.add_argument(
      "--output_path",
      type=str,
      default= "tfrecord",
      help="Directory where to store the output TFRecord files.")

   parser.add_argument(
      "--output_shards",
      type=int,
      default=1,
      help="Number of shards for the output.")

   FLAGS, unparsed = parser.parse_known_args()
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
