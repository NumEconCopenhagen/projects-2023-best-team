# Data analysis project

Our project is titled NASDAQ Index and ARIMA Model and is about the NASDAQ index closing values over the past 3 years. We use the yfinance library for historical data of the NASDAQ index, and processes that data to take the log differences of the Close variable. We then plot the close variable and the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF) of the log differences to determine the AR and MA parameters from the ARIMA model (Autoregressive Integrated Moving Average). We fit an ARIMA model to the log differences with the appropriate AR and MA parameters, and provide a summary of the results with the statsmodels library. 
We aim to provide insight into NADSAQ index behaviour and to forecast future changes. 

The **results** of the project can be seen from running [dataproject.ipynb](dataproject.ipynb).

We apply the **following datasets**:

1. dataX.csv (yahoo finance api) 
2. dataY.csv (yahoo finance api)

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

pip install matplotlib-venn
pip install yfinance
pip install y financials
pip install pandas
pip install statsmodels
pip install numpy
