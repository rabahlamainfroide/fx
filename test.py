
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
import pandas as pd
import numpy as np

p_wind_size=10
f_wind_size=10


file='csv_dir/EURUSD_M1.csv'
#infer_datetime_format, chunksize, print(df.loc[df['<DTYYYYMMDD>_<TIME>'].apply(datetime.minute)==0])

def minutes(datetime):
    return datetime.minute

def main():
    x=[0,0,0,0]
    y=[1,2,3,4]

    sum = np.vstack((x,y))
    print(np.sum(sum, axis=0))







if __name__ == '__main__':
    main()
