#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:59:01 2017

@author: yjiang
"""

from netCDF4 import Dataset
import numpy as np
from pandas import datetime
from pandas import read_csv
from scipy.stats.stats import pearsonr   
from matplotlib import pyplot
from numpy import unravel_index

def parser(x):
    if x.endswith('11') or x.endswith('12')or x.endswith('10'):
        return datetime.strptime(x, '%Y%m')
    else:
       return datetime.strptime(x, '%Y0%m') 
df = read_csv('preprocessed/indice_olr_excluded.csv', header=0, parse_dates=[0], 
              index_col=0, date_parser=parser)

lag = 6
for t in range(lag+1):
    soi = df.values[372+t:,0]      
    soi = soi.reshape(soi.shape[0], 1)

    nc = Dataset("/Users/yjiang/Downloads/sst.mnmean.nc", "r", format="NETCDF4")
    sst_full = np.array(nc.variables['sst'][:])
    sst = sst_full[1:-2-t,:,:]
    nc.close()

    r2 = []
    for i in range(sst.shape[1]):
        for j in range(sst.shape[2]):
            r2_index = pearsonr(soi, sst[:,i,j].reshape(sst.shape[0], 1))[0]
            r2.append(r2_index)

    r2_map = np.array(r2).reshape(sst.shape[1], sst.shape[2])
    max_index = unravel_index(r2_map.argmax(), r2_map.shape)

    pyplot.figure()
    pyplot.imshow(r2_map, extent=[-180, +180, -90, 90])
    pyplot.colorbar()
    
    r2_map = np.abs(r2_map)
    max_index = unravel_index(r2_map.argmax(), r2_map.shape)
    pyplot.xlabel('lag=' + str(t) + ', ' + str(max_index) + ', ' + str(r2_map[max_index]))
    # print(max_index, r2_map[max_index])
    

# good resources: http://pordlabs.ucsd.edu/cjiang/python.html
# http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/10_read_netcdf_python.pdf
# http://unidata.github.io/netcdf4-python/#netCDF4.Variable.getValue
# plot image: https://matplotlib.org/users/image_tutorial.html