# enso-forcasting

Traditional numerical simulations build upon human expertise in defining events based on subjective thresholds of relevant physical variables. Often, multiple competing methods produce vastly different results on the same dataset. This work explores using deep learning to improve the forecast accuracy of El Niño–Southern Oscillation (ENSO) based on time series data such as sea level pressure, ocean wind, and sea surface temperature.

Specifically, the code here includes a few modules: 
* Denoising via Inverse Fourier Analysis and Wavelet analysis
* Correlation between SOI index and a number of variables over time
* Lagged Linear Modeling
* RNN/LSTM modeling

## Current results:
Correlation between SOI and SST across all difference grids over time
![lag0](https://user-images.githubusercontent.com/5552606/34017287-282af5b8-e0f3-11e7-8a86-31e5addaa09c.png)

LSTM predictions result<br/>
<img width="501" alt="screen shot 2017-11-04 at 8 10 13 pm" src="https://user-images.githubusercontent.com/5552606/34017302-3862a2b4-e0f3-11e7-99b7-188bcb51c986.png">
