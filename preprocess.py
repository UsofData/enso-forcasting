#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:41:47 2017
Description: preprocess/reshape standard PSD format data
Note: 
1. need to convert the np array to float as some items would just be string
2. added an additional space before -99.90 manually
3. nino3 anomaly: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino3/
   climate indexes: https://www.esrl.noaa.gov/psd/data/climateindices/list/
@author: yjiang
"""
import numpy as np
import pandas as pd

# =============================================================================
# # oni preprocessing
# df=pd.read_csv('original_data/enso_oni.csv', sep=',')
# data = df.iloc[:,1:13].values
# =============================================================================

# change sep depending on the particular input file format
df=pd.read_csv('original_data/enso_nino3anomaly.txt', sep='   ', header=None)
data = df.iloc[:,1:13].values

print(type(data))
data_reshape = data.reshape(data.shape[0]*data.shape[1], 1)
data_reshape = data_reshape.astype(np.float32)
#==============================================================================
# np.set_printoptions(precision=2, suppress=True)
# print(data_reshape)
#==============================================================================


np.savetxt("preprocessed/enso_nino3anomaly_reshaped.csv", data_reshape, '%6.2f', delimiter=",")