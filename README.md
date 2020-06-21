# Forecasting concetration of PM10 particles based on weather data
Building a complete Machine Learning pipeline for forecasting air pollution 12 hours ahead.
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

**Feature selection** - creating statistical and seasonal features and dropping features that do not have effect on air pollution, 
concatenating datasets and handling different categorical features.

**Model selection** - For creating lag features previous 7 days were observed, for each hour separately, by the LSTM layer. Data is splitted, schuffled, scaled and reshaped to have proper shape for LSTM layer. The network is trained on Google colaboratory.

Loss:

![Loss](https://github.com/Data-Science-kosta/Forecasting-air-pollution-LSTM-model-and-Regression-trees-model/blob/master/data/model%20selected%20LSTM/plots/Loss.png)

Final evaluation:

![Loss](https://github.com/Data-Science-kosta/Forecasting-air-pollution-LSTM-model-and-Regression-trees-model/blob/master/data/model%20selected%20LSTM/plots/Final%20eval.png)
