#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:54:00 2017
Description: use the 6th last month of data to predict the current one
Reference: https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
@author: yjiang
"""

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_olr_excluded.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
print(df.head())

#check missing value
pd.isnull(df).any()

#draw a histogram and a table description
pyplot.figure(0)
df.ONI.hist()
df.describe()

#distribution over time
pyplot.figure(1)
pyplot.plot(df.ONI)
pyplot.title('ONI over Time')
pyplot.xlabel('Time')

#look at the correlation and heatmap
pyplot.figure(2)
df_cor = df.corr()
sns.heatmap(df_cor)


# Create lagged dataset
values = DataFrame(df.values)
df_lag = concat([values.shift(6), values.shift(5), values.shift(4), values.shift(3), values.shift(2), values.shift(1), values], axis=1)
#df_lag.columns = ['T-1_SOI', 'T-1_ONI','T_SOI', 'T_ONI']
print(df_lag.head())

# split into train and test sets
X = df_lag.values
train_size = int(len(X) * 0.66)
train, test = X[6:train_size+6], X[train_size+6:]
train_X, train_y = train[:,0:13], train[:,13]
test_X, test_y = test[:,0:13], test[:,13]

# persistence model
def model_persistence(x):
	return x[1]

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
pyplot.figure(3)
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
