#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:25:25 2017
1) seq2seq
@author: yjiang
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.layers import TimeDistributed


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

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m')

# load the dataset   
df = read_csv('preprocessed/indice_olr_excluded.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
values = df.values
values = values.astype('float32')

# specify the sliding window size and number of features
lag = 1
n_features = 2

# frame as supervised learning
reframed = series_to_supervised(values, lag, 1)
print(reframed.head())

# Define and Fit Model
rf_values = reframed.values
n_train = int(len(rf_values) * 0.7)
train = rf_values[:n_train, :]
test = rf_values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-2], train[:, -2:]
test_X, test_y = test[:, :-2], test[:, -2:]


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((1, train_X.shape[0], n_features))
test_X = test_X.reshape((1, test_X.shape[0], n_features))
train_y = train_y.reshape((1, train_y.shape[0], n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
batch = 1
model = Sequential()
model.add(LSTM(20, batch_input_shape = (batch, None, n_features), return_sequences=True, stateful=True))
model.add(LSTM(10, return_sequences=True, stateful=True))
model.add(LSTM(2, return_sequences=True, stateful=True))
# model.add(Dense(2))
model.compile(loss='mae', optimizer='adam')
# fit network
for i in range(0, 1):
    model.fit(train_X, train_y, batch_size = batch, epochs=1, verbose=2, shuffle=False)
	#model.train_on_batch(train_X, train_y, batch_size = 64)
    model.reset_states()

# prediction
next_X = model.predict(train_X)[0, -1, 0:n_features].reshape(1, 1, n_features)
new_seq = train_X
for i in range(0, test_X.shape[1]):
    new_seq = concatenate([new_seq, next_X], axis=1)
    next_X = model.predict(new_seq)[0, -1, 0:n_features].reshape(1, 1, n_features)
    print(i)

yhat = new_seq[:, n_train:, :].reshape(test_y.shape[0], n_features)
train_y = train_y.reshape(train_y.shape[1], n_features)
pyplot.plot(train_y[:, 1])
pyplot.plot([None for i in train_y[:, 1]] + [x for x in test_y[:, 1]])
pyplot.plot([None for i in train_y[:, 1]] + [x for x in yhat[:, 1]])
pyplot.show()

#==============================================================================
# # design network
# model = Sequential()
# model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
# 
# # evaluate the model
# # make a prediction
# yhat = model.predict(test_X)
# test_y = test_y.reshape((len(test_y), 1))
# # calculate RMSE
# rmse = sqrt(mean_squared_error(test_y, yhat))
# print('Test RMSE: %.3f' % rmse)
# 
# pyplot.plot(train_y)
# pyplot.plot([None for i in train_y] + [x for x in test_y])
# pyplot.plot([None for i in train_y] + [x for x in yhat])
# pyplot.show()
#==============================================================================
