import os
import numpy as np
import pandas as pd
from datetime import datetime
from function import forecast
import os
import sys
from load_data import load_data

# ---------------------- LOAD DATA
full_data, data_copy = load_data()

# ---------------------- PARAMETERS, TRAIN/VAL SETS AND SAVING FOLDER
# Hyperparameters
length = full_data.shape[0]
print(length)
cycle = 2  # 151 = int(11.2*(365/27)) i.e. number of Carrington rotations in one cycle
n_cycles = int(length / cycle)
percent_val = 0.15
n_neurons = [128, 0, 0]
batch_size = 16
epochs = 5
dropout = 0
learning_rate = 0.0005
version = 'shift_cycle'

# Train and validation datasets creation
X_train = np.zeros((n_cycles-1, cycle))  # Create a 2D NumPy array
y_train = np.zeros((n_cycles-1, cycle))  # Create a 2D NumPy array

for i in np.arange(n_cycles-1):
    X_train[i, :] = full_data.values[i*cycle:(i+1)*cycle, 0]
    y_train[i, :] = full_data.values[(i+1)*cycle:(i+2)*cycle, 0]

X_train = X_train.reshape((n_cycles-1, 1, cycle))  # Reshape to (n_cycles-1, 1, cycle)
y_train = y_train.reshape((n_cycles-1, 1, cycle))  # Reshape to (n_cycles-1, 1, cycle)
print(X_train.shape)
print(y_train.shape)
# Folder creation to save config and plots
folder_name = 'results_FC_shift_cycle'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
folder = folder_name
if not os.path.exists(folder):
    os.mkdir(folder)
file = open(folder+'/config.txt', 'w')
file.write('cycle: ' + str(cycle))
file.write('\nn_neurons: ' + str(n_neurons))
file.write('\nbatch_size: ' + str(batch_size))
file.write('\nepochs: ' + str(epochs))
file.write('\ndropout: ' + str(dropout))
file.write('\nlr: ' + str(learning_rate))
file.write('\nactivation: ' + 'ReLU')
file.write('\noptimizer: ' + 'Adam')

# ---------------------- FORECAST
forecast(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)