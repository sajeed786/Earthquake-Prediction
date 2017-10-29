import numpy as np
import pandas as pd

from Feature_4_slope import read_feature_4
from filter import read_data, path_list, threshold

paths = path_list()


def read_feature_6(path):
    df = read_data(path)

    df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
    new_df = df.loc[df["Magnitude"] > threshold]
    x = new_df['Date']
    new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
    del new_df["Time"]
    groups = new_df.groupby(
        new_df.Date.dt.month)  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    months_available = groups.groups.keys()  # https://stackoverflow.com/questions/28844535/python-pandas-groupby-get-list-of-groups
    feat = read_feature_4(path)
    M_observed = []
    for month in months_available:
        M_observed.append(groups.get_group(month)["Magnitude"].max())
    M_expected = []
    for a, b in feat:
        if (b == 0.0 or a == 0.0):
            M_expected.append(0.0)
        else:
            M_expected.append(a / b)
    deficit = np.subtract(np.array(M_observed), np.array(M_expected))
    # print(deficit)
    return deficit


def read_full_feature_6():
    magnitude_deficit = []
    for path in paths:
        magnitude_deficit.append(read_feature_6(path))
    return np.array(magnitude_deficit)


read_feature_6(paths[1])
