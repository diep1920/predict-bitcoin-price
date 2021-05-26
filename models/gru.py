from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout

class GRUNet:
    def arch(self, n_layers, hidden_size, dropout_rate):
        n_timesteps = 50
        n_features = 8

        model = Sequential()
        model.add(GRU(units=hidden_size, return_sequences=True, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(dropout_rate))
        for i in range(n_layers - 2):
            model.add(GRU(units=hidden_size, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(GRU(units=hidden_size))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=n_features))

        model.compile(loss="mse", optimizer="adam")

        return model
