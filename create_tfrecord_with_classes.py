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
import threading




class time_series (object):
    def __init__(self, df,p_wind_size,f_wind_size):
        self.df = df
        self.p_wind_size=p_wind_size
        self.f_wind_size=f_wind_size
        self.offset=0
        self.window_discarded=0
        self.lapswindow=timedelta(minutes=p_wind_size+f_wind_size-1)


    def window(self,frame, tolerance):

        def check_window(frame, tolerance):
            for i in range (self.df.shape[0]-self.offset-self.p_wind_size-self.f_wind_size):
                first = self.df.iloc[self.offset,0]
                last = self.df.iloc[self.offset +self.p_wind_size+ self.f_wind_size-1,0]
                lapse=last-first
                if (lapse - timedelta(minutes=(self.p_wind_size+ self.f_wind_size-1) * frame) > timedelta(minutes=tolerance))  :
                    self.window_discarded +=1
                    self.offset += 1
                    if (self.offset +self.f_wind_size>= self.df.shape[0]):
                        return False
                else:
                    return True
        #if not check_window(frame, tolerance):
        #    return None,None, None, None, None, None


        p_wind= np.array(self.df.iloc[self.offset:self.offset + self.p_wind_size,2:6])
        f_wind=np.array(self.df.iloc[self.offset + self.p_wind_size-1:self.offset +self.p_wind_size+ self.f_wind_size-1,2:6])
        time_date = self.df.iloc[self.offset + self.p_wind_size,0]
        vol = self.df.iloc[self.offset + self.p_wind_size,6]
        self.offset=self.offset+1
        if (self.offset +self.f_wind_size>= self.df.shape[0]):
            return False
        return np.rot90(p_wind), np.rot90(f_wind), time_date.hour, time_date.weekday(),time_date.day,vol


class label_generator(object):
    def __init__(self,profit, loss):
        self.sell_counter = 0
        self.loss= loss/10000
        self.profit=profit/10000
        self.buy_counter = 0
        self.neutral_counter = 0

    def get_label(self,f_wind, f_wind_size):

        lowest_low=np.amin(f_wind[1])
        highst_high=np.amax(f_wind[3])
        close_price=f_wind[1,f_wind_size-2]
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


def print_recurssevely(str):
        print(str, end='')
        print('\r', end='')
        time.sleep(0.2)


def chunk_to_tfrecord_classes(df_chunk,total_data, p_wind_size,f_wind_size, profit, loss, output_path, num_classes = 3, frame=5, tolerance=10):
    writers = []
    for i in range(num_classes):
      writers.append(tf.python_io.TFRecordWriter(output_path+'_class_' + str(i)))
    a = label_generator(profit, loss)
    chunk =df_chunk.loc[df_chunk['<DTYYYYMMDD>_<TIME>'].apply(minutes) % frame==0]
    tm = time_series(chunk,p_wind_size,f_wind_size)
    for i in range(df_chunk.shape[0]-tm.p_wind_size-tm.f_wind_size-10):
        p_wind, f_wind, hour, day_week, day_month,vol  = tm.window(frame, tolerance)
        if (vol==None): break
        label=a.get_label(f_wind, f_wind_size)
        features = {}
        features["p_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=p_wind.flatten()))
        features["f_wind"] = tf.train.Feature(float_list=tf.train.FloatList(value=f_wind.flatten()))
        features["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        features["day_week"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_week]))
        features["day_month"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[day_month]))
        features["hour"]= tf.train.Feature(int64_list=tf.train.Int64List(value=[hour]))
        f = tf.train.Features(feature=features)
        example = tf.train.Example(features=f)
        writers[label].write(example.SerializeToString())
        #print_recurssevely(str(int(tm.offset)*15/total_data ) + ' %')
    for w in writers:
        w.close()



def minutes(datetime):
    return datetime.minute

def multithreading_csv_to_tfrecord(reader, total_data, chunk_size,  output_path, p_wind_size, f_wind_size, profit, loss):
  coord = tf.train.Coordinator()
  threads = []
  for i in range(3):
        df_chunk = reader.get_chunk(chunk_size)
        args = (df_chunk,total_data, p_wind_size,f_wind_size, profit, loss, output_path +'_thread_' + str(i))
        t = threading.Thread(target=chunk_to_tfrecord_classes, args=args)
        t.start()
        threads.append(t)


  coord.join(threads)
  sys.stdout.flush()


def main(argv):

    output_name = FLAGS.pair +'_'+ FLAGS.frame +'_'+ str(FLAGS.pwind) +'_'+ str(FLAGS.fwind) +'_'+ str(FLAGS.loss) +'_'+ str(FLAGS.profit)
    output_path = os.path.join(FLAGS.output_path, output_name)
    csv_name = FLAGS.pair +'_'+ FLAGS.frame
    csv_patern = os.path.join(FLAGS.csv_dir, csv_name)
    file_handles = glob.glob(csv_patern + '*.csv')
    total_data =  len(open(file_handles[0]).readlines())
    chunk_size = int(total_data/3)
    reader = pd.read_csv(file_handles[0], parse_dates=[[1,2]], infer_datetime_format= True , delimiter = ',', iterator=True)

    multithreading_csv_to_tfrecord(reader,total_data,chunk_size, output_path,  FLAGS.pwind, FLAGS.fwind, FLAGS.profit, FLAGS.loss)














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
