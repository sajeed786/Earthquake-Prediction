import pandas as pd

threshold = 4.0
file_name = "database.csv"
Dates = []

def get_Dates():
    global file_name
    global Dates
    df = pd.read_csv(file_name)
    df = df[df.Type == "Earthquake"]
    df = df[df.Magnitude > threshold]
    df.Date = df.Date + ' '  + df.Time
    del df["Time"]
    del df["Type"]

    # Grouping Data Into Months
    df.Magnitude = df.Magnitude.apply(pd.to_numeric)
    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %H:%M:%S')
    Dates = df.Date
    print(Dates)
    return Dates


def read_data():
    global file_name
    df = pd.read_csv(file_name)
    df = df[df.Type == "Earthquake"]
    df = df[df.Magnitude > threshold]
    df.Date = df.Date + ' '  + df.Time
    del df["Time"]
    del df["Type"]

    # Grouping Data Into Months
    df.Magnitude = df.Magnitude.apply(pd.to_numeric)
    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %H:%M:%S')
    groups = df.groupby(df.Date.dt.year)
    return groups

def read_frame():
    global file_name
    df = pd.read_csv(file_name)
    df = df[df.Type == "Earthquake"]
    df.Date = df.Date + ' '  + df.Time
    del df["Time"]
    del df["Type"]

    # Grouping Data Into Months
    df.Magnitude = df.Magnitude.apply(pd.to_numeric)
    df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %H:%M:%S')
    return df
