#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:41:47 2017

@author: yjiang
"""
import numpy as np
import pandas as pd
df=pd.read_csv('original_data/enso_oni.csv', sep=',')
data = df.iloc[:,1:13].values
print(type(data))
data_reshape = data.reshape(data.shape[0]*data.shape[1], 1)
#==============================================================================
# np.set_printoptions(precision=2, suppress=True)
# print(data_reshape)
#==============================================================================
np.savetxt("preprocessed/enso_oni_reshaped.csv", data_reshape, '%6.2f', delimiter=",")