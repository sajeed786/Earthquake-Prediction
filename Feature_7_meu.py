import numpy as np
import pandas as pd
from Read_Data import read_frame

def read_feature_7():
    df = read_frame()
    lower_threshold = 7.0
    upper_threshold = 7.5
    df = df[df.Magnitude > lower_threshold]
    df = df[df.Magnitude <= upper_threshold]
    df = df.groupby(df.Date.dt.year)
    Meu = []
    years = df.groups.keys()
    for year in years:
        new_df = df.get_group(year)
        new_df = new_df.groupby(new_df.Date.dt.month)
        months_available = new_df.groups.keys()
        indi = [0]*12
        for month in months_available:
            ddf = new_df.get_group(month).Date
            ddf = ddf.tolist()
            ti = []
            n_char = len(ddf)
            prev = ddf[0]
            for i in range(1,len(ddf)):
                curr = ddf[i]
                ti.append(np.float64(((curr - prev)/ np.timedelta64(1, 's'))/86400))
            indi[month-1] = np.sum(np.array(ti))/n_char
        Meu.append(indi)
    dictionary = dict(zip(years, Meu))
    return dictionary

# df = read_feature_7()
# print(pd.DataFrame(df))