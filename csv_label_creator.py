
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




csv_pattern = "csv_dir/EURUSD_M1"

file_handles = glob.glob(csv_pattern + '*.csv')

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

    a = label_generator(profit, loss)

    max_data=10000 #(df.shape[0]-tm.p_wind_size-tm.f_wind_size)
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

        df(create new column = label)
        if (label == 0): df[offset,"label"] = 0
        if (label == 1): df[offset,"label"] = 0
        if (label == 2): df[offset,"label"] = 0


    print('offset =', tm.offset,"window_discarded = ", tm.window_discarded )
    print("buy = ", a.buy_counter, "sell =  ", a.sell_counter, "neutral =  ", a.neutral_counter)




def main(argv):
    p_wind_size =20
    f_wind_size =10


    df = pd.read_csv(file_handle, delimiter = ';')
    df.loc[df[''] == some_value]
    time_date = df.iloc[,0]
    time_date= parse(time_date)

    print(file_handles[i], " processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
