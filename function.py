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
        # Initialize X_train_forecast with your own values (adjust as needed)
        input_data = np.array([
            75.5,75.6   # Add your values here
        ])

        # Should have the same length as the cycle
        cycle_length = 2

        # Check if the input data length matches the cycle
        if len(input_data) != cycle_length:
            raise ValueError("Input data length should match the cycle length.")

        # Reshape the input to match the model's input shape
        X_train_forecast = input_data.reshape(1, 1, cycle_length)

        # Perform the shifting and prediction as before
        forecasts = []
        for _ in np.arange(cycle_length):
            current_forecasts = model.predict(X_train_forecast)
            to_add = current_forecasts[0][-1]
            forecasts.append(to_add[0])  # Convert to_add to a scalar using [0]

            # Shift values to the left and append the new value
            X_train_forecast = np.roll(X_train_forecast, shift=-1, axis=2)
            X_train_forecast[0, 0, -1] = to_add[0]  # Convert to_add to a scalar using [0]

        # get the result after prediction
        print(forecasts)
        
    elif version == 'shift_cycle':# Define the input data for prediction (adjust the values as needed)
        input_data = np.array([
            75.5,75.6 
        ])
        # Should have 151 values
        cycle_length = 2

        # Check if the input data length matches the cycle
        if len(input_data) != cycle_length:
            raise ValueError("Input data length should match the cycle length.")

        # Reshape the input to match the model's input shape
        input_data = input_data.reshape(1, 1, cycle_length)

        # Make predictions using the model
        forecasts = model.predict(input_data).reshape(-1)
        
        # get the result after prediction
        print(forecasts)
        
    model.save(folder+'/model')
    
    
