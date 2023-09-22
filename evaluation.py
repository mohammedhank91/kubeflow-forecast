import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import ReLU
import sys
import time
from predicition import prediction


def evaluation(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version):
    forecasts = prediction(X_train, y_train, batch_size, epochs, n_neurons,
                           learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
 # Loss on the train set
    train_mse = model.evaluate(X_train, y_train)
    for name, values in history.history.items():
        plt.plot(values, label=name)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(fontsize="14")
        plt.title('Train RMSE: '+str(round(train_mse, 2)), size=18)
    plt.savefig(folder + '/loss.png')

    # We take the first data as the first value of the first cycle, so the cycle that
    # we want to predict is already started and we know some on its first values
    already_known = length-n_cycles*int(cycle)

    # Forecast plot
    # last day of the forecasted cycle (knowing that 1 cycle = 151 values)
    last_forecast = datetime(year=2030, month=11, day=1)
    time_range = pd.date_range(
        start=full_data.index[0], end=last_forecast, freq='27D').tolist()
    noaa_pred = pd.read_csv('dataScraped/F107_monthly_predicted.csv', sep=';')
    noaa_pred['epoch'] = pd.to_datetime(noaa_pred['epoch'])
    noaa_pred = noaa_pred[noaa_pred['epoch'] < last_forecast]
    plt.figure(figsize=(10, 6))
    plt.plot(time_range[:length], full_data.values,
             color='royalblue', label='observed')
    plt.plot(time_range[length-already_known:], forecasts,
             color='mediumseagreen', label='forecast')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_min'],
             color='coral', label='noaa F10.7 min')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_moy'],
             color='indianred', label='noaa F10.7 mean')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_max'],
             color='firebrick', label='noaa F10.7 max')

    plt.plot()
    plt.xlabel('time')
    plt.title("F10.7 (Fully connected neural network)")
    plt.legend()
    plt.savefig(folder + '/forecast.png')

    return forecasts
