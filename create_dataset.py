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

wind_size = 20
f_wind_size= 10
profit= 5
loss =5
buy=5
sell=0
neutral=0

def _pick_output_shard(output_shards):
  return np.random.randint(0, output_shards - 1)


def convert_csv(csv_dir , output_file, output_shards):
    file_handles = []
    for filename in sorted(tf.gfile.ListDirectory(csv_dir)):
      if not filename.endswith(".csv"):
        print("Skipping", filename)
        continue
      file_handles.append(tf.gfile.GFile(os.path.join(csv_dir, filename), "r"))


    writers = []
    for i in range(output_shards):
     writers.append(
         tf.python_io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i,
                                                          output_shards)))


    for file_handle in file_handles :
        df = pd.read_csv(file_handle, delimiter = ';')
        tm = time_series(df,wind_size,f_wind_size)
        for f in file_handles:
            tfrecord_generator(df,tm, writers, output_shards)
            f.close()

    for w in writers:
       w.close()





class time_series (object):
    def __init__(self, df,wind_size,f_wind_size):
        self.df = df
        self.wind_size=wind_size
        self.f_wind_size=f_wind_size
        self.offset=0
        self.window_discarded=0
        self.lapswindow=timedelta(minutes=wind_size+f_wind_size-1)

    def window(self):

        def check_window(tolerance=3):
            for i in range (self.df.shape[0]-self.offset-self.wind_size-self.f_wind_size):
                first = self.df.iloc[self.offset,0]
                last = self.df.iloc[self.offset +self.wind_size+ self.f_wind_size-1,0]
                lapse=parse(last)-parse(first)
                if (lapse - timedelta(minutes=self.wind_size+ self.f_wind_size-1) > timedelta(minutes=tolerance))  :
                    self.window_discarded +=1
                    self.offset += 1
                    if (self.offset +self.f_wind_size>= self.df.shape[0]):
                        return False
                else:
                    return True
        if not check_window():
            return None,None, None, None, None
        p_wind= np.array(self.df.iloc[self.offset:self.offset + self.wind_size,1:5])
        f_wind=np.array(self.df.iloc[self.offset + self.wind_size-1:self.offset +self.wind_size+ self.f_wind_size-1,1:5])
        time_date = self.df.iloc[self.offset + self.wind_size,0]
        time_date= parse(time_date)
        self.offset=self.offset+1
        if (self.offset +self.f_wind_size>= self.df.shape[0]):
            return False
        return np.transpose(p_wind), np.transpose(f_wind),time_date.hour, time_date.weekday(),time_date.day


class label_generator(object):
    def __init__(self, profit, loss):
        self.sell = 0
        self.loss= loss/10000
        self.profit=profit/10000
        self.buy = 0
        self.neutral = 0

    def get_label(self,fw):

        lowest_low=np.amin(fw[0])
        highst_high=np.amax(fw[1])
        close_price=fw[2,f_wind_size-1]
        open_price=fw[3,0]
        up_swing=highst_high - open_price
        down_swing=open_price - lowest_low
        if (up_swing >= self.profit) and (down_swing <=self.loss) :
            label = 1
            self.buy+=1                        #:
        else :
            if (up_swing <= self.loss)  and (down_swing >=self.profit):
                label = 2
                self.sell+=1
            else :
                label=0
                self.neutral+=1
        return label

a = label_generator(profit,loss)


def print_recurssevely(str):
        print(str, end='')
        print('\r', end='')
        time.sleep(0.2)

def tfrecord_generator(df,tm,writers,output_shards):

    max_data=10000 #(df.shape[0]-tm.wind_size-tm.f_wind_size)
    for i in range(max_data):
        if (i % 1000 == 0) :
            print_recurssevely(str(int(i * 100/max_data)) + ' %')
        day_week=None
        i=0
        while day_week is None:
            i+=1
            p_wind, f_wind, hour, day_week, day_month  = tm.window()


            if (df.shape[0] - tm.offset - tm.wind_size - tm.f_wind_size) <=0 :
                print("end of file reached.  offset =", tm.offset,"window_discarded = ", tm.window_discarded, )
                return 0

        label=a.get_label(f_wind)
        print(p_wind)
        '''


        features = {}
        features["f_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=f_wind.flatten()))
        features["p_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=p_wind.flatten()))
        features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        features["day_week"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_week]))
        features["day_month"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_month]))
        features["hour"]= tf.train.Feature(int64_list=tf.train.Int64List(value=[hour]))
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)

        writers[_pick_output_shard(output_shards)].write(example.SerializeToString())
        '''

    print('offset =', tm.offset,"window_discarded = ", tm.window_discarded )
    print("buy = ", a.buy, "sell =  ", a.sell, "neutral =  ", a.neutral)

def main(argv):
  del argv

  convert_csv(FLAGS.csv_dir, FLAGS.output_path, FLAGS.output_shards)



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--csv_dir",
      type=str,
      default="csv_dir",
      help="Directory where the ndjson files are stored.")
  parser.add_argument(
      "--output_path",
      type=str,
      default= "tfrecord/tfrecord",
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
