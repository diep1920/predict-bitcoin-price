from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten

class Conv1DNet:
    def arch(self, n_layers, n_filters, kernel_size, dropout_rate):
        n_timesteps = 50
        n_features = 8

        model = Sequential()
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(MaxPooling1D(pool_size=2))
        for i in range(n_layers - 1):
            model.add(Conv1D(filters=n_filters, kernel_size=kernel_size, activation='relu'))
            model.add(Dropout(dropout_rate))
            model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(units=n_features))

        model.compile(loss="mse", optimizer="adam")

        return model