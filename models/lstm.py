from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

class LSTMNet:
    def arch(self, n_layers, hidden_size, dropout_rate):
        n_timesteps = 50
        n_features = 8

        model = Sequential()
        model.add(LSTM(units=hidden_size, return_sequences=True, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(dropout_rate))
        for i in range(n_layers - 2):
            model.add(LSTM(units=hidden_size, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(units=hidden_size))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=n_features))

        model.compile(loss="mse", optimizer="adam")

        return model
