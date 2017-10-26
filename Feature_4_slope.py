import numpy as np
import pandas as pd

from filter import read_data, path_list, threshold

paths = path_list()


def read_feature_4(path):
    df = read_data(path)

    df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
    new_df = df.loc[df["Magnitude"] > threshold]
    x = new_df['Date']
    new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
    del new_df["Time"]
    groups = new_df.groupby(
        new_df.Date.dt.month)  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    months_available = groups.groups.keys()  # https://stackoverflow.com/questions/28844535/python-pandas-groupby-get-list-of-groups


def read_full_feature_4():
    slope = []
    for path in paths:
        slope.append(read_feature_4(path))
    return np.array(slope)
