import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

frame = pd.read_csv("titanic.csv")
print(frame.head())
x = frame
x.drop(["Name", "Sex"], axis= 1, inplace=True)
x.Age.fillna(x.Age.mean(), inplace=True)
print(x.Age.isnull().sum())
print(x.head())

# minmmax = MinMaxScaler(feature_range=(0, 1))
# minmmax.fit_transform