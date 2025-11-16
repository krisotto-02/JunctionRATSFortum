README

ELECTRICITY CONSUMPTION FORECASTING

This project forecasts electricity consumption for multiple groups of customers using SARIMAX models in Python. It produces:
- 48-hour energy consumption forecast
- 12-month energy consumption forecast

DEPENDENCIES
The following Python libraries were used in the Project, you can install them through pip:
- pathlib
- typing
- pandas 
- __future__
- pickle
- statsmodels

FUNCTIONALITY AND STRUCTURE
>loadData: Handles data loading, reads hourly consumption and prices from the training Excel files and CSVs for the 48h and 12m time horizons.
>dataProcessing: Preprocesses the loaded training data and aligns power consumption information with prices, builds calendar features (hour, weekday, month), and constructs future exogenous data for model training.
>train48Hours and train12Months: Trains each customer on a SARIMAX machine learning model for both time periods, based on seasonal data and electricity prices
>forecast48Hours and forecast12Months: Implements forecasting functionality for the next 48 hours or 12 months
>converter: Converts the forecasted data into requested format and outputs it to a CSV file
>main: Puts the whole Program together

OPERATING THIS PROJECT
> In order to receive the forecasts, you must run the main file
> The main file contains certain parameters you can tune in the end.
> do_train: Set this either to args.skip_training or Not args.skip_training, whether you want to train new models for client groups or not
> train_days_48h: Set this to a hour time-period of your choice or args.train_days_48h for max amount of training hours
> train_months_12m: Set this to a month time-period of your choice or args.train_months_12m for max amount of training months
> max_groups: Amount of client Group you want to train, either set a number, or leave as args.max_groups
> Run the file --> the forecasts should appear in the respective Data folder


