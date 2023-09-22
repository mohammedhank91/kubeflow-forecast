import os
import pandas as pd
from datetime import datetime
import sys
import time
from training import train_model
from forecast import forecast
from prediction import prediction
from load_data import load_data

def save_result():
    # ---------------------- LOAD DATA 
    full_data, data_copy = load_data() 
    # ---------------------- SAVE FORECAST IN CSV FILE
    epoch = pd.DataFrame(time_range[length-already_known:], columns=['epoch'])

    df_forecast = pd.DataFrame(forecasts, columns=['value'])
    df_forecast = pd.concat([epoch, df_forecast], axis=1)
    to_save = pd.concat([data_copy, df_forecast[already_known:]], axis=0)
    to_save.to_csv(folder+'/forecast_F107.csv', sep=';', index=False)
