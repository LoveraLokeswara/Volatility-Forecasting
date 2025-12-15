# Volatility Forecasting

A comprehensive study of hybrid models for financial volatility forecasting, combining traditional econometric methods with machine learning approaches.

## Overview

This project explores various hybrid forecasting models applied to financial time series data (S&P 500, Bitcoin, EUR/USD). It compares traditional statistical methods (ARIMA, GARCH) with machine learning techniques (LSTM, SVM) and their combinations.

## Hybridization Methods

### Method 1: Additive Assumption
- **ARIMA-LSTM Hybrid**: Combines ARIMA for linear patterns with LSTM for non-linear residuals
- **ARIMA-SVM Hybrid**: Uses ARIMA for trend forecasting and SVM for residual prediction

### Method 2: Non-Additive Assumption
- **ARIMA-LSTM Hybrid**: LSTM uses ARIMA predictions as additional input features
- **ARIMA-SVM Hybrid**: SVM uses ARIMA predictions as additional input features
- **GARCH-LSTM Hybrid**: LSTM uses GARCH volatility predictions as additional input features

## Models Included

- **ARIMA**: AutoRegressive Integrated Moving Average
- **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity
- **LSTM**: Long Short-Term Memory neural networks
- **SVM**: Support Vector Machines

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: pandas, numpy, tensorflow, scikit-learn, statsmodels, arch, yfinance

## Usage

Each Jupyter notebook is self-contained and can be run independently:
- Individual model analysis: `ARIMA_LSTM_SVM.ipynb`
- Hybrid models in their respective folders
