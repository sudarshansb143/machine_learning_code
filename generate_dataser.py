import datetime
import math
import pandas as pd
import random
import radar
from faker import Faker
import numpy as np
import matplotlib as plt
fake = Faker()

def generateData(n):
  listdata = []
  start = datetime.datetime(2019, 8, 1)
  end = datetime.datetime(2019, 8, 30)
  delta = end - start
  for _ in range(n):
    date = radar.random_datetime(start='2019-08-1', stop='2019-08-30').strftime("%Y-%m-%d")
    price = round(random.uniform(900, 1000), 4)
    listdata.append([date, price])
  df = pd.DataFrame(listdata, columns = ['Date', 'Price'])
  df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
  df = df.groupby(by='Date').mean()

  return df

frame = generateData(50)
print(frame.head())

import seaborn as sns
sns.pairplot(frame)
plt.show()