import pandas as pd

from filter import read_data

pd.options.mode.chained_assignment = None  # default='warn'

df = read_data()
threshold = 4.0

df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
new_df = df.loc[df["Magnitude"] > threshold]
# x = new_df['Date']+' '+new_df['Time']
# new_df["Datetime"] = pd.to_datetime(x ,format='%Y/%m/%d %H:%M:%S.%f')
x = new_df['Date']
new_df["Date"] = pd.to_datetime(x, format='%Y/%m/%d')
del new_df["Time"]
groups = new_df.groupby(new_df.Date.dt.month).agg({'Date': ['first',
                                                            'last']})  # Problem Solved By link:- https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/

print(groups.Date)
