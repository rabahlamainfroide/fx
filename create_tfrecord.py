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

def _pick_output_shard(output_shards):
  if (output_shards == 1):
    return 0
  else :
    return np.random.randint(0, output_shards - 1)

def convert_csv(csv_patern , output_path, output_shards,p_wind_size, f_wind_size, profit, loss):
    file_handles = glob.glob(csv_patern + '*.csv')
    writers = []
    for i in range(output_shards):
      writers.append(
         tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_path, i,
                                                          output_shards)))
    for i in range(len(file_handles)) :
        df = pd.read_csv(file_handles[i], delimiter = ';')
        tm = time_series(df,p_wind_size,f_wind_size)
        tfrecord_generator(df,tm, writers, output_shards,f_wind_size, profit, loss)
        print(file_handles[i], " processed")
    for w in writers:
        w.close()


class time_series (object):
    def __init__(self, df,p_wind_size,f_wind_size):
        self.df = df
        self.p_wind_size=p_wind_size
        self.f_wind_size=f_wind_size
        self.offset=0
        self.window_discarded=0
        self.lapswindow=timedelta(minutes=p_wind_size+f_wind_size-1)

    def window(self):

        def check_window(tolerance=3):
            for i in range (self.df.shape[0]-self.offset-self.p_wind_size-self.f_wind_size):
                first = self.df.iloc[self.offset,0]
                last = self.df.iloc[self.offset +self.p_wind_size+ self.f_wind_size-1,0]
                lapse=parse(last)-parse(first)
                if (lapse - timedelta(minutes=self.p_wind_size+ self.f_wind_size-1) > timedelta(minutes=tolerance))  :
                    self.window_discarded +=1
                    self.offset += 1
                    if (self.offset +self.f_wind_size>= self.df.shape[0]):
                        return False
                else:
                    return True
        if not check_window():
            return None,None, None, None, None
        p_wind= np.array(self.df.iloc[self.offset:self.offset + self.p_wind_size,1:5])
        f_wind=np.array(self.df.iloc[self.offset + self.p_wind_size-1:self.offset +self.p_wind_size+ self.f_wind_size-1,1:5])
        time_date = self.df.iloc[self.offset + self.p_wind_size,0]
        time_date= parse(time_date)
        self.offset=self.offset+1
        if (self.offset +self.f_wind_size>= self.df.shape[0]):
            return False
        return np.transpose(p_wind), np.transpose(f_wind),time_date.hour, time_date.weekday(),time_date.day


class label_generator(object):
    def __init__(self,profit, loss):
        self.sell_counter = 0
        self.loss= loss/10000
        self.profit=profit/10000
        self.buy_counter = 0
        self.neutral_counter = 0

    def get_label(self,f_wind, f_wind_size):

        lowest_low=np.amin(f_wind[0])
        highst_high=np.amax(f_wind[1])
        close_price=f_wind[2,f_wind_size-1]
        open_price=f_wind[3,0]
        up_swing=highst_high - open_price
        down_swing=open_price - lowest_low
        if (up_swing >= self.profit) and (down_swing <=self.loss) :
            label = 1
            self.buy_counter+=1                        #:
        else :
            if (up_swing <= self.loss)  and (down_swing >=self.profit):
                label = 2
                self.sell_counter+=1
            else :
                label=0
                self.neutral_counter+=1
        return label


def print_recurssevely(str):
        print(str, end='')
        print('\r', end='')
        time.sleep(0.2)

def tfrecord_generator(df,tm,writers,output_shards,f_wind_size, profit, loss):
    a = label_generator(profit, loss)

    max_data=1 #(df.shape[0]-tm.p_wind_size-tm.f_wind_size)
    for i in range(max_data):
        if (i % 1000 == 0) :
            print_recurssevely(str(int(i * 100/max_data)) + ' %')
        day_week=None
        i=0
        while day_week is None:
            i+=1
            p_wind, f_wind, hour, day_week, day_month  = tm.window()


            if (df.shape[0] - tm.offset - tm.p_wind_size - tm.f_wind_size) <=0 :
                print("end of file reached.  offset =", tm.offset,"window_discarded = ", tm.window_discarded, )
                return 0

        label=a.get_label(f_wind, f_wind_size)
        f_wind = np.fliplr(f_wind)

        features = {}
        features["p_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=p_wind.flatten()))
        features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        features["day_week"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_week]))
        features["day_month"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_month]))
        features["hour"]= tf.train.Feature(int64_list=tf.train.Int64List(value=[hour]))
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)

        writers[_pick_output_shard(output_shards)].write(example.SerializeToString())

    print('offset =', tm.offset,"window_discarded = ", tm.window_discarded )
    print("buy = ", a.buy_counter, "sell =  ", a.sell_counter, "neutral =  ", a.neutral_counter)

def main(argv):

    output_name = FLAGS.pair +'_'+ FLAGS.frame +'_'+ str(FLAGS.pwind) +'_'+ str(FLAGS.fwind) +'_'+ str(FLAGS.loss) +'_'+ str(FLAGS.profit)
    output_path = os.path.join(FLAGS.output_path, output_name)
    csv_name = FLAGS.pair +'_'+ FLAGS.frame
    csv_patern = os.path.join(FLAGS.csv_dir, csv_name)

    convert_csv(csv_patern, output_path, FLAGS.output_shards, FLAGS.pwind, FLAGS.fwind, FLAGS.profit, FLAGS.loss)


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument(
       "--pair",
       type=str,
       default="EURUSD",
       help="Pair")
   parser.add_argument(
       "--frame",
       type=str,
       default="M1",
       help="Frame")
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
      "--train_observations_per_class",
      type=int,
      default=10000,
      help="How many items per class to load for training.")
   parser.add_argument(
      "--eval_observations_per_class",
      type=int,
      default=1000,
      help="How many items per class to load for evaluation.")
   parser.add_argument(
      "--output_shards",
      type=int,
      default=10,
      help="Number of shards for the output.")

   FLAGS, unparsed = parser.parse_known_args()
   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
