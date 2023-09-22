import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.keras.layers import ReLU

def train_model(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version):

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
    return model, history
