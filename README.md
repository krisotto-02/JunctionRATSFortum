# Electricity Consumption Forecasting (48h & 12m)
This project forecasts electricity consumption for multiple customer groups using **SARIMAX** models in Python. It produces:
- **48-hour forecasts** 
- **12-month forecasts**
Outputs are formatted to match the example CSVs from the Fortum/Junction challenge.
## Project Structure
JunctionRATS/
├─ Data/
│  ├─ 20251111_JUNCTION_training.xlsx
│  ├─ 20251111_JUNCTION_example_hourly.csv
│  ├─ 20251111_JUNCTION_example_monthly.csv
│  └─ models/
│     ├─ sarimax_48h/      # saved hourly models (per group)
│     └─ sarimax_12m/      # saved monthly models (per group)
└─ src/
   ├─ loadData.py          # load training + example data
   ├─ dataProcessing.py    # feature engineering, exogenous builders
   ├─ train48Hours.py      # train SARIMAX models for 48h horizon
   ├─ train12Months.py     # train SARIMAX models for 12m horizon
   ├─ forecast48Hours.py   # generate 48h forecasts
   ├─ forecast12Months.py  # generate 12m forecasts
   ├─ converter.py         # convert forecasts to submission CSVs
   ├─ evaluation.py        # MAPE + FVA% vs naive baselines (UNUSED)
   └─ main.py              # pipeline entry point

**Functionality**
> Data loading
loadData.py reads:
hourly consumption and prices from the training Excel file
example CSVs for the 48h and 12m horizons (for timestamps and column layout)

>Preprocessing & features
dataProcessing.py:
aligns consumption with prices,
builds calendar features (hour of day, weekday/weekend, month, etc.),
constructs future exogenous data (calendar + prices) for the forecast horizons.

>Model training
train48Hours.py:
trains per-group SARIMAX models on recent hourly data for the 48h forecast.
train12Months.py:
trains per-group SARIMAX models on monthly aggregates for the 12m forecast.

Models are saved under:
Data/models/sarimax_48h/
Data/models/sarimax_12m/

> Forecasting & submissions
forecast48Hours.py and forecast12Months.py load the trained models and generate forecasts for all groups.
converter.py reshapes these forecasts to match the example CSV formats and writes submission-ready files.
