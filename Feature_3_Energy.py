import numpy as np
import pandas as pd

from Feature_1_time import read_feature_1
from filter import read_data, path_list, threshold

paths = path_list()

def read_feature_3(path):
    df = read_data(path)

    df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
    new_df = df.loc[df["Magnitude"] > threshold]
    x = new_df['Date']
    new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
    del new_df["Time"]
    groups = new_df.groupby(new_df.Date.dt.month)  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
    T = read_feature_1(path)
    months_available = groups.groups.keys() # https://stackoverflow.com/questions/28844535/python-pandas-groupby-get-list-of-groups
    dE = []
    # Solved using https://pandas.pydata.org/pandas-docs/stable/groupby.html        - (get_group() attribute)
    for month in months_available:
        dE.append(np.sum(groups.get_group(month)["Magnitude"].apply(lambda x: np.sqrt(10**(11.8+1.5*x)))))
    dE = np.array(dE)

    send = []
    for i in range(0,len(T)):
        if T[i] == 0:
            continue
        #print("dE: {},T: {},dE/T: {}".format(dE[i],T[i],dE[i]/T[i]))
        send.append(dE[i]/T[i])
    return np.array(send)

def read_full_feature_3():
    Energy = []
    for path in paths:
        Energy.append(read_feature_3(path))
    return np.array(Energy)