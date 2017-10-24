import pandas as pd


def read_data():
    df = pd.read_csv("SCEC_DC\\1932.catalog", skiprows=10, header=None,
                     sep=r"\s*",
                     engine='python')  # Solved Problem For Reading Multile Spaces link:- https://stackoverflow.com/questions/12021730/can-pandas-handle-variable-length-whitespace-as-column-delimiters

    df.drop(df.tail(3).index,
            inplace=True)  # drop last 3 rows because of useless data Link:- https://stackoverflow.com/questions/26921651/how-to-delete-the-last-row-of-data-of-a-pandas-dataframe

    df = df.rename(columns={0: "Date"})
    df = df.rename(columns={1: "Time"})
    df = df.rename(columns={2: "ET"})
    df = df.rename(columns={3: "GT"})
    df = df.rename(columns={4: "Magnitude"})
    df = df.rename(columns={5: "M"})
    df = df.rename(columns={6: "Lat"})
    df = df.rename(columns={7: "Lon"})
    df = df.rename(columns={8: "Depth"})
    df = df.rename(columns={9: "Q"})
    df = df.rename(columns={10: "EVID"})
    df = df.rename(columns={11: "NPH"})
    df = df.rename(columns={12: "NGRM"})

    # print(df.head())

    return df
