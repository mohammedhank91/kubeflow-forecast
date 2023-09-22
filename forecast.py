import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import ReLU
import os
import sys
import time


def forecast(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version):
    model = Sequential()
    model.add(Input(shape=(1, cycle)))
    model.add(Dense(n_neurons[0], activation=ReLU()))
    # model.add(Dense(n_neurons[1], activation=ReLU()))
    # model.add(Dense(n_neurons[2], activation=ReLU()))
    model.add(Dense(cycle, activation="linear"))
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=learning_rate),
    )

    model.summary()
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        shuffle=True,
    )

    if version == 'simple_shift':
        # Take the last value of the forecasts of the last 151 (cycle) values, drop the first value and repeat the process with the new last value
        forecasts = []
        X_train_forecast = np.zeros((1, 1, cycle))

        for _ in np.arange(cycle):
            current_forecasts = model.predict(X_train_forecast)
            to_add = current_forecasts[0][-1]
            forecasts.append(to_add[0])  # Convert to_add to a scalar using [0]

            # Shift values to the left and append the new value
            X_train_forecast = np.roll(X_train_forecast, shift=-1, axis=2)
            X_train_forecast[0, 0, -1] = to_add[0]  # Convert to_add to a scalar using [0]

    elif version == 'shift_cycle':
        forecasts = model.predict(full_data.values[(n_cycles - 1) * cycle:n_cycles * cycle].reshape(1, 1, cycle)).reshape(cycle, )

    model.save(folder+'/model')

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
    last_forecast = datetime(year=2030, month=11, day=1)  # last day of the forecasted cycle (knowing that 1 cycle = 151 values)
    time_range = pd.date_range(start=full_data.index[0], end=last_forecast, freq='27D').tolist()
    noaa_pred = pd.read_csv('dataScraped/F107_monthly_predicted.csv', sep=';')
    noaa_pred['epoch'] = pd.to_datetime(noaa_pred['epoch'])
    noaa_pred = noaa_pred[noaa_pred['epoch'] < last_forecast]
    plt.figure(figsize=(10, 6))
    plt.plot(time_range[:length], full_data.values, color='royalblue', label='observed')
    plt.plot(time_range[length-already_known:], forecasts, color='mediumseagreen', label='forecast')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_min'], color='coral', label='noaa F10.7 min')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_moy'], color='indianred', label='noaa F10.7 mean')
    plt.plot(noaa_pred['epoch'], noaa_pred['value_max'], color='firebrick', label='noaa F10.7 max')

    plt.plot()
    plt.xlabel('time')
    plt.title("F10.7 (Fully connected neural network)")
    plt.legend()
    plt.savefig(folder + '/forecast.png')

    # ---------------------- SAVE FORECAST IN CSV FILE
    epoch = pd.DataFrame(time_range[length-already_known:], columns=['epoch'])

    df_forecast = pd.DataFrame(forecasts, columns=['value'])
    df_forecast = pd.concat([epoch, df_forecast], axis=1)
    to_save = pd.concat([data_copy, df_forecast[already_known:]], axis=0)
    to_save.to_csv(folder+'/forecast_F107.csv', sep=';', index=False)
