import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from function import forecast
import sys
from load_data import load_data

# ---------------------- LOAD DATA
full_data, data_copy = load_data()

# ---------------------- PARAMETERS, TRAIN/VAL SETS AND SAVING FOLDER
# Hyperparameters
length = full_data.shape[0]
cycle = 2  # 151 = int(11.2*(365/27)) i.e. number of Carrington rotations in one cycle
n_cycles = int(length / cycle)
percent_val = 0.15
n_neurons = [128, 0, 0]
batch_size = 16
epochs = 5
dropout = 0
learning_rate = 0.0005
version = 'simple_shift'

# Train and validation datasets creation
X_train = pd.DataFrame(np.zeros([length-cycle, cycle]))
y_train = pd.DataFrame(np.zeros([length-cycle, cycle]))
for i in np.arange(y_train.shape[0]):
    X_train.iloc[i, :] = full_data.values[i:i + cycle, 0]
    y_train.iloc[i, :] = full_data.values[i+1:i+1+cycle, 0]

# Reshape X_train and y_train to match the model's input shape
X_train = X_train.values.reshape(-1, 1, cycle)
y_train = y_train.values.reshape(-1, 1, cycle)

val_size = int(X_train.shape[0] * percent_val)
rand_list = random.sample(range(0, X_train.shape[0]), val_size)

# Remove .values when indexing X_train
X_val = X_train[rand_list]
y_val = y_train[rand_list]

# Remove .values when dropping rows from X_train
X_train = np.delete(X_train, rand_list, axis=0)
y_train = np.delete(y_train, rand_list, axis=0)

# Folder creation to save config and plots
folder_name = 'results_FC_simple_shift'
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
file.write('\nlearning_rate: ' + str(learning_rate))
file.write('\nactivation: ' + 'ReLU')
file.write('\noptimizer: ' + 'Adam')

# ---------------------- FORECAST
forecast(X_train, y_train, batch_size, epochs, n_neurons, learning_rate, cycle, n_cycles, full_data, length, data_copy, folder, version)
