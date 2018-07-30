import pandas as pd
import numpy as np
from Read_Data import Dates,get_Dates
from Feature_1_time import read_feature_1
from Feature_2_Mean_Magnitude import read_feature_2
from Feature_3_Energy import read_feature_3
from Feature_4_slope import read_feature_4
from Feature_5_deviation import read_feature_5
from Feature_6_magnitude_deficit import read_feature_6
from Feature_7_meu import read_feature_7

print('>>> Import Completed\n >>> Reading Feature 1')
Feature_1 = read_feature_1()
print('>>> Feature 1 Read Complete. \n >>> Reading Feature 2')
Feature_2 = read_feature_2()
print('>>> Feature 2 Read Complete. \n >>> Reading Feature 3')
Feature_3 = read_feature_3()
print('>>> Feature 3 Read Complete. \n >>> Reading Feature 4')
Feature_4 = read_feature_4()
print('>>> Feature 4 Read Complete. \n >>> Reading Feature 5')
Feature_5 = read_feature_5()
print('>>> Feature 5 Read Complete. \n >>> Reading Feature 6')
Feature_6 = read_feature_6()
print('>>> Feature 6 Read Complete. \n >>> Reading Feature 7')
Feature_7 = read_feature_7()
print('>>> Feature 7 Read Complete. \n >>> Converting To Dataframe...')

Final_df = []
for (_,i),(_,j),(_,k),(_,l),(_,m),(_,n),(_,o) in zip(Feature_1.items(),Feature_2.items(),Feature_3.items(),Feature_4.items(),Feature_5.items(),Feature_6.items(),Feature_7.items()):
    for a,b,c,d,e,f,g in zip(i,j,k,l[1],m,n,o):
        Final_df.append((a,b,c,d,e,f,g))

Final_df = pd.DataFrame(Final_df)
Final_df.to_pickle("Dataset.pkl")
Final_df.to_csv("Dataset.csv")