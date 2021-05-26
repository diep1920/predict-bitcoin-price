import pandas as pd
import numpy as np
import sys

data_file_name = sys.argv[1]
dataset = pd.read_csv('./dataset/' + data_file_name + '.csv')

# Separate into train set (80% of dataset) and test set (20%)
n_train_rows = int(dataset.shape[0]*.8)-1
train = dataset.iloc[:n_train_rows, :].values
test = dataset.iloc[n_train_rows:, :].values

# Prepare the training data
x_train = []
y_train = []

steps = 50

for i in range(steps, train.shape[0]):
    x_train.append(train[i-steps:i, :])
    y_train.append(train[i, :])

x_train, y_train = np.array(x_train), np.array(y_train)
np.save('./dataset/x_train.npy', x_train)
np.save('./dataset/y_train.npy', y_train)

# Prepare the test data
x_test = []
y_test = []

for i in range(steps, test.shape[0]):
    x_test.append(test[i-steps:i, :])
    y_test.append(test[i, :])

x_test, y_test = np.array(x_test), np.array(y_test)
np.save('./dataset/x_test.npy', x_test)
np.save('./dataset/y_test.npy', y_test)