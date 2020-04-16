# Forecasting concetration of PM10 particles based on weather data
Building a complete Machine Learning pipeline for forecasting pollution 12 hours ahead.
## DATA:
The data is collected from 8 different stations in Macedonia.
Links from where the data was collected:
https://pulse.eco/restapi
https://darksky.net/dev/docs
## MODELS: 
**2** Models were built. In the first model each station has its own predictor, and in the second model (LSTM model) datasets are concatenated and universal predictor is made.
### 1. model:
**Preprocessing** - removing or interpolating missing values, transforming categorical data, dropping redundant features.
**Feature selection** - creating lag, seasonal and statistical features and dropping features that do not have effect on air pollution.
**Model selection** - 3 models were built for each station (Linear regressor, Extra treees regressor, XGBoost regressor)
### 2. model:
**Preprocessing** - removing or interpolating missing values, transforming categorical data, dropping redundant features.
**Feature selection** - creating statistical and seasonal features and dropping features that do not have effect on air pollution, concatenating datasets and handling different categorical features.
**Model selection** - For creating lag features past 7 days were observed, for each hour separately, by the LSTM layer. Data is splitted, schuffled, scaled and reshaped to have proper shape for LSTM layer. The network is trained on Google colaboratory.

Loss:
![Loss](https://github.com/666KostA666/Data-Science/blob/master/Forecasting%20air%20pollution%20(LSTM%20model%20and%20Regression%20trees%20model)/data/model%20selected%20LSTM/plots/Loss.png)
Final evaluation:
![Loss](https://github.com/666KostA666/Data-Science/blob/master/Forecasting%20air%20pollution%20(LSTM%20model%20and%20Regression%20trees%20model)/data/model%20selected%20LSTM/plots/Final%20eval.png)
