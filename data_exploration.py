#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 11:24:56 2017

@author: yjiang
"""

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
from pandas import datetime
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_everything_included.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
print(df.head())

#check missing value
pd.isnull(df).any()

# remove column olr
df = df.drop('olr', 1)

# remove the first few rows and start from 1979-1-1; if select columns: dataframe.iloc[:,0:10]
# https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
start = 336 
df = df.iloc[start:]

# normalize a specific column
# df['precip'] = (df['precip'] - df['precip'].mean()) / (df['precip'].max() - df['precip'].min())

# standadize all columns
# differencing: df's built-in function - https://stackoverflow.com/questions/19939896/diff-on-pandas-dataframe-with-more-than-one-column
# manually: https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/
df = (df - df.mean()) / df.std()

# df.plot()
# df['soi'].plot() # or df.soi.plot()
# plot each column

# values = df.values
i = 1
fig = pyplot.figure()
for col in df.columns.tolist():
    fig.add_subplot(len(df.columns.tolist()), 1, i)
    df[col].plot()
    # pyplot.plot(values[:, i-1])
    pyplot.title(col, y=0.8, loc='right')
    if i != len(df.columns.tolist()):
    # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
        pyplot.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        # pyplot.xticks([], [])
        pyplot.xlabel('')
    i += 1
pyplot.show()

# fft: https://plot.ly/matplotlib/fft/
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.fft.fftfreq.html
soi = df['soi'].values
n = len(soi)
Y = np.fft.fft(soi)/n
Y = Y[range(int(n/2))]

timestep = 1.0
frq = np.fft.fftfreq(n, d=timestep)
frq = frq[range(int(n/2))]

fig, ax = pyplot.subplots(2, 1)
ax[0].plot(soi)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('soi')
ax[1].plot(frq, abs(Y), 'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

pyplot.figure();
autocorrelation_plot(df.soi)
pyplot.title('soi autocorrelation chart')
# pyplot.show()

pyplot.figure()
df_cor = df.corr()
sns.heatmap(df_cor)


