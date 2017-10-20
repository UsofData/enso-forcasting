#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:55:22 2017
Reference: 1) https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
2) extension: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/
@author: yjiang
"""

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
#from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_olr_excluded.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

#==============================================================================
# autocorrelation_plot(df.ONI)
# pyplot.title('ONI autocorrelation chart')
# pyplot.show()
#==============================================================================

X = df.ONI
size = int(len(X) * 0.95)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(10,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test.values)
pyplot.plot(predictions, color='red')
pyplot.show()