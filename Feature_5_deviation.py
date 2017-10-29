import math
from collections import Counter

import numpy as np
import pandas as pd

from filter import read_data, path_list, threshold

paths = path_list()


def read_feature_5(path):
    df = read_data(path)

    df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
    new_df = df.loc[df["Magnitude"] > threshold]
    x = new_df['Date']
    new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
    del new_df["Time"]
    groups = new_df.groupby(
        new_df.Date.dt.month)  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    months_available = groups.groups.keys()  # https://stackoverflow.com/questions/28844535/python-pandas-groupby-get-list-of-groups
    lst = []
    for month in months_available:
        sorted_mag = groups.get_group(month)["Magnitude"]
        sorted_mag = np.sort(sorted_mag)
        n = len(sorted_mag)
        mapped_mag = dict(Counter(sorted_mag))
        M = []
        N = []
        for mag in mapped_mag:
            M.append(mag)
            N.append(n)
            n = n - mapped_mag[mag]
        N = list(map(math.log, N))
        n = len(sorted_mag)
        part1 = np.sum(np.array(M) * np.array(N))
        part2 = np.sum(np.array(M)) * np.sum(np.array(N))
        part3 = np.square(np.sum(np.array(M)))
        part4 = np.sum(np.square(np.array(M)))
        b = 0.0
        eta = 0.0
        if n == 1:
            b = 0.0
            a = np.sum(np.array(N)) / n
            eta = 0.0
        else:
            b = (n * part1 - part2) / (part3 - n * part4)
            M = b * np.array(M)
            a = np.sum(np.array(N) + np.array(M)) / n
            eta = np.sum(np.square(np.array(N) - (a - M))) / (n - 1)
        lst.append(eta)
    # print(lst)
    return np.array(lst)


def read_full_feature_5():
    deviation = []
    for path in paths:
        deviation.append(read_feature_5(path))
    return np.array(deviation)
