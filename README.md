# ALY-6140 Capstone Project

## Project Overview

This project uses the M5 Walmart sales dataset to forecast daily demand for food products at the product-store level. The goal is to analyze historical food sales patterns and build forecasting models.


## Research Question

How accurately can food demand be forecasted at the product-store level using historical sales, calendar features, price information, SNAP indicators, event variables, and product-store identifiers?

## Project Structure

```text
Capstone Project/
│
├── notebooks/
│   ├── 01_Clean data.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature engineering.ipynb
│   └── 04_Modeling.ipynb
│
├── src/
│   ├── calendar_features.py
│   ├── evaluate_prophet.py
│   ├── evaluation_index.py
│   ├── event_feature.py
│   ├── lag_features.py
│   ├── lgb_predict.py
│   ├── m5_utils.py
│   ├── price_feature.py
│   └── rolling_features.py
```


## Data

The original dataset comes from the Kaggle M5 Forecasting - Accuracy competition. Due to file size and data redistribution considerations, the raw and processed data files are not included in this repository.

To reproduce the project, download the M5 dataset from Kaggle and place the following files in the project folder or the appropriate data folder:

* calendar.csv
* sales_train_evaluation.csv
* sales_test_evaluation.csv
* sell_prices.csv

## Methods

The project includes the following steps:

1. Data cleaning and preparation
2. Exploratory data analysis
3. Feature engineering
4. Forecasting model development
5. Model evaluation and comparison

The main models used in this project are:

* Prophet
* LightGBM

Prophet is used as an interpretable time series benchmark model. LightGBM is used as the main machine learning model because it can use lag features, rolling statistics, price features, calendar variables, SNAP indicators, event variables, and product-store identifiers.

## Evaluation Metrics

Model performance is evaluated using:

* RMSSE
* MAE

RMSSE is useful for comparing forecast accuracy across product-store series with different sales scales. MAE is included because it is easy to interpret as the average absolute sales error.

## Key Findings

The exploratory analysis shows that food demand contains trend, weekly seasonality, monthly seasonality, price effects, SNAP effects, event effects, and strong store-level differences. The data also contains many intermittent demand series with a high number of zero-sales days.

LightGBM performs better than Prophet on the representative evaluation subset because it can use more engineered features and learn patterns across multiple product-store series.







