import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.rnn import SimpleRNNNet
from models.lstm import LSTMNet
from models.gru import GRUNet
from models.conv1d import Conv1DNet
from keras.models import load_model
import sys
import os


# change hyperparameters here
rnn_n_layers, rnn_hidden_size, rnn_dropout_rate = 3, 50, 0.2
cnn_n_layers, cnn_n_filters, cnn_kernel_size, cnn_dropout_rate  = 2, 60, 3, 0.3

models = [LSTMNet, GRUNet, SimpleRNNNet, Conv1DNet][int(sys.argv[1])]
n_steps = 50
n_epochs = 10

x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')

# Normalize features to avoid overfitting
scaler = MinMaxScaler(feature_range = (0, 1))
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
y_train_scaled = scaler.fit_transform(y_train)
x_test_scaled = scaler.fit_transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
y_test_scaled = scaler.fit_transform(y_test)

model_file_name = './model_files/' + models.__name__ + '_' + str(n_steps) + 'steps_' + str(n_epochs) + '_epochs.h5'
if os.path.isfile(model_file_name):
    model = load_model(model_file_name)
else:
    if models.__name__ == "Conv1DNet":
        model = models().arch(n_layers=cnn_n_layers,
                              n_filters=cnn_n_filters,
                              kernel_size=cnn_kernel_size,
                              dropout_rate=cnn_dropout_rate)
    else:
        model = models().arch(n_layers=rnn_n_layers,
                              hidden_size=rnn_hidden_size,
                              dropout_rate=rnn_dropout_rate)
    model.fit(x_train_scaled, y_train_scaled, batch_size=32, epochs=n_epochs)
    model.save(model_file_name)
    print("Saved model")

print("Evaluate on test data")
loss = model.evaluate(x_test_scaled, y_test_scaled, batch_size=32)
print("test loss: " +  str(loss))

y_hat = model.predict(x_test_scaled)
y_hat = scaler.inverse_transform(y_hat)

ASK_PRICE = 0
BID_PRICE = 3
# Visualise the ask_price predictions
plt.figure(figsize = (18,9))
plt.plot(y_test[:,ASK_PRICE], color = 'red', label = 'y_test')
plt.plot(y_hat[:,ASK_PRICE], color = 'blue', label = 'y_hat')
plt.title('y_hat["ask_price"] vs y_test["ask_price"]. Loss = ' + str(loss))
plt.ylabel('ask_price')
plt.legend()
plt.savefig('./figure/predict/' + models.__name__ + '_ask_price_prediction.png', bbox_inches='tight')

# Visualise the bid_price predictions
plt.figure(figsize = (18,9))
plt.plot(y_test[:,BID_PRICE], color = 'red', label = 'y_test')
plt.plot(y_hat[:,BID_PRICE], color = 'blue', label = 'y_hat')
plt.title('y_hat["bid_price"] vs y_test["bid_price"]. Loss = ' + str(loss))
plt.ylabel('bid_price')
plt.legend()
plt.savefig('./figure/predict/' + models.__name__ + '_bid_price_prediction.png', bbox_inches='tight')