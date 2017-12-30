#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:42:38 2017
Reference: 
1) multiple time step: https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
2) multiple variate&lag: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
@author: yjiang
"""

from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


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
        cols.append(df.shift(-i).iloc[:,-1])
        if i == 0:
            names += ['VAR(t)']
        else:
            names += ['VAR(t+%d)' % i]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# fit a linear model
def fit_linear(train, n_ahead):
    # reshape training into [samples, timesteps, features]
    X, Y = train[:, :-n_ahead], train[:, -n_ahead:]
    regr = linear_model.LinearRegression()
    model = MultiOutputRegressor(regr).fit(X, Y)
    return model

# fit a linear model
def fit_randomF(train, n_ahead):
    # reshape training into [samples, timesteps, features]
    X, Y = train[:, :-n_ahead], train[:, -n_ahead:]
    regr = RandomForestRegressor(max_depth = 30, random_state=2)
    model = MultiOutputRegressor(regr).fit(X, Y)
    return model

def forecast_linear(model, X):
    # make forecast
    forecast = model.predict(X)
    # convert to array
    return [x for x in forecast[0, :]]

def make_forecasts(model, test, n_ahead):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, :-n_ahead]
        # make forecast
        forecast = forecast_linear(model, X)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(y, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in y]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        
# plot the forecasts in the context of the original dataset, multiple segments
def plot_forecasts(series, forecasts, n_test, xlim, ylim, n_ahead, linestyle = None):
    # plot the entire dataset in blue
    pyplot.figure()
    if linestyle==None:
        pyplot.plot(series, label='observed')
    else:
        pyplot.plot(series, linestyle, label='observed')
    pyplot.xlim(xlim, ylim)
    pyplot.legend(loc='upper right')
    # plot the forecasts in red
    for i in range(len(forecasts)):
        if i%n_ahead ==0: # this ensures not all segements are plotted, instead it is plotted every n_ahead
               off_s = len(series) - n_test + 2 + i - 1
               off_e = off_s + len(forecasts[i]) + 1
               xaxis = [x for x in range(off_s, off_e)]
               yaxis = [series[off_s]] + forecasts[i] 
               pyplot.plot(xaxis, yaxis, 'r')
               # print(off_s, off_e)
    # show the plot
    pyplot.show()
        
def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_everything_included.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)

df = df.drop('olr', 1)
start = 336 
df = df.iloc[start:]
df = (df - df.mean()) / df.std()

cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]

enso = df.values.astype('float32')

# specify the sliding window size and number of features
lag = 12
ahead = 3
n_features = 1
# frame as supervised learning
reframed = series_to_supervised(enso, lag, ahead)
# drop columns we don't want to predict
print(reframed.head())

# Define and Fit Model
values = reframed.values
n_train = int(len(values) * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]

# fit model

# model = fit_linear(train, ahead)
model = fit_randomF(train, ahead)

forecasts = make_forecasts(model, test, ahead)

# evaluate forecasts
actual = [row[-ahead:] for row in test]
evaluate_forecasts(actual, forecasts, lag, ahead)

# plot forecasts
plot_forecasts(df['soi'].values, forecasts, test.shape[0] + ahead - 1, 0, 500, ahead)
plot_forecasts(df['soi'].values, forecasts, test.shape[0] + ahead - 1, 360, 470, ahead, 'go')


