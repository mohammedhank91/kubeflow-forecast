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
from training import train_model

def prediction(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version):
    
    model, history = train_model(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
    
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
    
    return forecasts