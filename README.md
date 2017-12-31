# enso-forcasting

Traditional numerical simulations build upon human expertise in defining events based on subjective thresholds of relevant physical variables. Often, multiple competing methods produce vastly different results on the same dataset. This work explores using deep learning to improve the forecast accuracy of El Niño–Southern Oscillation (ENSO) based on time series data such as sea level pressure, ocean wind, and sea surface temperature.

Specifically, the code here includes a few modules: 
* Pattern analysis via Fast Fourier Transformation, Wavelet analysis, and autocorrelation
* Correlation between SOI index and a number of variables with different lags
* Compared the results of RNN/LSTM, multiple output linear regression, and random forest
