#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:42:38 2017
Reference: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
1) multiple time step
@author: yjiang
"""

from math import sqrt
import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# fit an LSTM network to training data
# The end of the epoch is the end of the sequence and the internal state should not 
# carry over to the start of the sequence on the next epoch.
# I run the epochs manually to give fine grained control over when resets occur (by 
# default they occur at the end of each batch).
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # y = y.reshape(y.shape[0], 1, y.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons[0], batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
#==============================================================================
#     # need to look into the multi-layer LSTM problem (input dimension)
#     model.add(LSTM(n_neurons[1], stateful=True))
#     model.add(LSTM(n_neurons[2], stateful=True))
#     model.add(LSTM(n_neurons[3], stateful=True))
#==============================================================================
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
        model.reset_states()
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

def make_forecasts(model, n_batch, train, test, n_lag):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, 0:n_lag]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(y, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in y]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test, xlim, ylim):
    # plot the entire dataset in blue
    pyplot.plot(series)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        if i%12 ==0:
               off_s = len(series) - n_test + 2 + i - 1
               off_e = off_s + len(forecasts[i]) + 1
               xaxis = [x for x in range(off_s, off_e)]
               yaxis = [series[off_s]] + forecasts[i] 
               pyplot.plot(xaxis, yaxis, color='red')
               pyplot.xlim(xlim, ylim)
               print(off_s, off_e)
    # show the plot
    pyplot.show()
        
def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_olr_excluded.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

v = df.values
ONI = v[:,1].reshape(v.shape[0],1)

# ensure all data is float
ONI = ONI.astype('float32')
# specify the sliding window size and number of features
lag = 120
ahead = 12
n_features = 1
# frame as supervised learning
reframed = series_to_supervised(ONI, lag, ahead)
# drop columns we don't want to predict
print(reframed.head())

# Define and Fit Model
values = reframed.values
n_train = int(len(values) * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]

# fit model
model = fit_lstm(train, lag, ahead, 1, 500, [30, 20, 10, 5])

forecasts = make_forecasts(model, 1, train, test, lag)

# evaluate forecasts
actual = [row[lag:] for row in test]
evaluate_forecasts(actual, forecasts, lag, ahead)
# plot forecasts
plot_forecasts(ONI, forecasts, test.shape[0] + ahead - 1, 620, 820)


