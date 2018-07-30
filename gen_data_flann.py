from datetime import datetime
import math
import numpy as np
import copy
import pandas as pd
import csv
from itertools import zip_longest
class Dataset:
    @staticmethod
    def load_from_file(filename):
        """
        Load and return data from file
        :param filename: path of the database.csv file
        :return: (date, latitude, longitude, magnitude) (np.array)
        """
        date, latitude, longitude, magnitude, depth = [], [], [], [], []
        df = pd.read_csv(filename, engine='python')
        df['Magnitude'] = df['Magnitude'].apply(pd.to_numeric)
        df['Latitude'] = df['Latitude'].apply(pd.to_numeric)
        df['Longitude'] = df['Longitude'].apply(pd.to_numeric)
        df['Depth'] = df['Depth'].apply(pd.to_numeric)
        df['Date'] = df['Date'] + df['Time']
        del df['Time']
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y%H:%M:%S')
        date = df['Date'].tolist()
        latitude = df['Latitude'].tolist()
        longitude = df['Longitude'].tolist()
        depth = df['Depth'].tolist()
        magnitude = df['Magnitude'].tolist()
        return np.array(date), np.float32(latitude), np.float32(longitude), np.float32(depth), np.float32(magnitude)

    @staticmethod
    def normalize_date(array):
        """
        Normalize datetime array
        :param array: array to normalize
        :return: normalized array (np.array)
        """
        min_data = min(array)
        max_data = max(array)
        delta = max_data - min_data
        return np.float32([(d - min_data).total_seconds() / delta.total_seconds() for d in array])

    @staticmethod
    def normalize_cord_rad(latitude, longitude):
        """
        Normalize GPS cord array, assuming the earth is shpherical
        :param latitude: latitude array to normalize
        :param longitude: longitude array to normalize
        :return: normalized arrays (np.array)
        """
        rad_lat = np.deg2rad(latitude)
        rad_lon = np.deg2rad(longitude)

        x = np.cos(rad_lat) * np.cos(rad_lon)
        y = np.cos(rad_lat) * np.sin(rad_lon)
        z = np.sin(rad_lat)

        return x, y, z
    @staticmethod
    def normalize_mag_depth(magnitude,depth):
        """
        normalize the magnitude in the range [0.1,0.9]
        """
        minm = min(magnitude)
        maxm = max(magnitude)
        diff = maxm - minm
        mag = 0.8*((magnitude - minm)/diff) + 0.1

        minm=min(depth)
        maxm=max(depth)
        delta=maxm-minm
        dep=(depth - minm)/delta
        
        return mag, dep
    @staticmethod
    def normalize_cord(latitude,longitude,lat_z):
        """
        normalize latitude and longitude in the range [0 1]
        """
        min1=min(latitude)
        max1=max(latitude)
        del1=max1-min1

        min2=min(longitude)
        max2=max(longitude)
        del2=max2-min2

        min3=min(lat_z)
        max3=max(lat_z)
        del3=max3-min3

        x = (latitude - min1)/del1
        y = (longitude - min2)/del2
        z = (lat_z - min3)/del3

        return x, y, z


if __name__ == "__main__":

    date, latitude, longitude, depth, magnitude = Dataset.load_from_file("database_original.csv")
    data_size = len(date)
    date = Dataset.normalize_date(date)
    latitude,longitude,lat_z = Dataset.normalize_cord_rad(latitude, longitude)
    latitude,longitude,lat_z = Dataset.normalize_cord(latitude,longitude,lat_z)
    mag_rev,depth = Dataset.normalize_mag_depth(magnitude,depth)
    mag_rev = mag_rev.tolist()
    date = date.tolist()
    latitude = latitude.tolist()
    longitude = longitude.tolist()
    lat_z = lat_z.tolist()
    depth = depth.tolist()
    magnitude = magnitude.tolist();
    d = [date, latitude, longitude, lat_z, depth, mag_rev, magnitude]
    export_data = zip_longest(*d, fillvalue = '')
    with open('quake_flls_norm.csv', 'w', newline='') as mf:
        wr = csv.writer(mf)
        wr.writerows(export_data)
    mf.close()
