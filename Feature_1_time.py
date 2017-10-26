import numpy as np
import pandas as pd

from filter import read_data, path_list

threshold = 4.0
paths = path_list()


def read_feature_1():
    for path in paths:
        df = read_data(path)

        df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
        new_df = df.loc[df["Magnitude"] > threshold]
        x = new_df['Date'] + ' ' + new_df['Time']
        new_df["Datetime"] = pd.to_datetime(x, format='%Y/%m/%d %H:%M:%S.%f')
        # x = new_df['Date']
        # new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
        del new_df["Time"]
        groups = new_df.groupby(new_df.Datetime.dt.month).agg({'Datetime': ['first',
                                                                            'last']})  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

        days = (groups.Datetime['last'] - groups.Datetime['first'])
        groups['seconds'] = days / np.timedelta64(1, 's')
        groups['days'] = days
        return days
