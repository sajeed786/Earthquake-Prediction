import pandas as pd

from filter import read_data, path_list

threshold = 4.0
paths = path_list()
oop = paths
for path in oop:
    df = read_data(path)

    df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
    new_df = df.loc[df["Magnitude"] > threshold]
    # x = new_df['Date']+' '+new_df['Time']
    # new_df["Datetime"] = pd.to_datetime(x ,format='%Y/%m/%d %H:%M:%S.%f')
    x = new_df['Date']
    new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
    del new_df["Time"]
    groups = new_df.groupby(
        new_df.Date.dt.month)  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

    print(groups['Magnitude'].mean())
