import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from models.rnn import SimpleRNNNet
from models.lstm import LSTMNet
from models.gru import GRUNet
from models.conv1d import Conv1DNet

# change hyperparameters candidates here
rnn_n_layers = [3, 4, 5]
#rnn_hidden_size = [40, 50, 60]
rnn_hidden_size = [50]
#rnn_dropout_rate = [0.2, 0.3, 0.4]
rnn_dropout_rate = [0.2]
cnn_n_layers = [2, 3]
#cnn_n_filters = [55, 60, 65]
cnn_n_filters = [60]
#cnn_kernel_size = [3, 5]
cnn_kernel_size = [3]
#cnn_dropout_rate = [0.3, 0.4, 0.5]
cnn_dropout_rate = [0.3]

rnn_configs = []
for n_layers in rnn_n_layers:
    for hidden_size in rnn_hidden_size:
        for dropout_rate in rnn_dropout_rate:
            rnn_configs.append([n_layers,
                                hidden_size,
                                dropout_rate])

cnn_configs = []
for n_layers in cnn_n_layers:
    for n_filters in cnn_n_filters:
        for kernel_size in cnn_kernel_size:
            for dropout_rate in cnn_dropout_rate:
                cnn_configs.append([n_layers,
                                    n_filters,
                                    kernel_size,
                                    dropout_rate])

# For future use of manually picking hyper parameter
cnn_configs_arr = np.array(cnn_configs)
table = np.concatenate((np.arange(0, cnn_configs_arr.shape[0]).reshape(1, cnn_configs_arr.shape[0]).T, cnn_configs_arr), axis=1)
np.savetxt("./figure/cnn_config_ref.csv", table, "%d,%d,%d,%d,%1.1f",header="id,n_layers,n_filters,kernel_size,dropout_rate")
rnn_configs_arr = np.array(rnn_configs)
table = np.concatenate((np.arange(0, rnn_configs_arr.shape[0]).reshape(1, rnn_configs_arr.shape[0]).T, rnn_configs_arr), axis=1)
np.savetxt("./figure/rnn_config_ref.csv", table, "%d,%d,%d,%1.1f", header="id,n_layers,hidden_size,dropout_rate")
####################################################

x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')

# Normalize features to avoid overfitting
scaler = MinMaxScaler(feature_range = (0, 1))
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
y_train_scaled = scaler.fit_transform(y_train)

n_folds = 5
n_epochs = 10

kfold = KFold(n_splits=n_folds, shuffle=True)

model_instants = {}
# +(model_instants)
# -----------+LSTMNet
# ----------------------+LSTMNet model with 3 layers, hidden size = 40, dropout rate = 0.2
# ----------------------+LSTMNet model with 3 layers, hidden size = 40, dropout rate = 0.3
# ----------------------...
# -----------+GRUNet
# ----------------------+GRUNet model with 3 layers, hidden size = 40, dropout rate = 0.2
# ----------------------...
# -----------+SimpleRNNNet
# ----------------------+SimpleRNNNet model with 3 layers, hidden size = 40, dropout rate = 0.2
# ----------------------...
# -----------+Conv1DNet
# ----------------------+Conv1DNet model with 2 layers, 55 filters, kernel size = 3, dropout rate = 0.3
# ----------------------...

best = {}
# +(best)
# -----------+LSTMNet
# ----------------------+loss: lowest loss value archived with LSTMNet architecture
# ----------------------+config: index of hyper-parameter configuration which gives lowest loss
# -----------+GRUNet
# ----------------------+loss
# ----------------------+config
# -----------+SimpleRNNNet
# ----------------------+loss
# ----------------------+config
# -----------+Conv1DNet
# ----------------------+loss
# ----------------------+config

logging = {}
# +(model_instants)
# -----------+LSTMNet
# ----------------------+index of hyper-parameter configuration 3 layers, hidden size = 40, dropout rate = 0.2
#----------------------------------+mean_loss: mean of loss values across 5-folds which are given by LSTMNet model
# ---------------------------------------------with 3 layers, hidden size = 40, dropout rate = 0.2
#----------------------------------+std_loss: standard deviation of loss values across 5-folds which are given by
# ---------------------------------------------LSTMNet model with 3 layers, hidden size = 40, dropout rate = 0.2
# ----------------------+index of hyper-parameter configuration 3 layers, hidden size = 40, dropout rate = 0.3
# ---------------------------------+(same as above)
# -----------+GRUNet (same as above)
# -----------+SimpleRNNNet (same as above)
# -----------+Conv1DNet (same as above)

# initiate data structures
models = [LSTMNet, GRUNet, SimpleRNNNet, Conv1DNet]
for model in models:
    logging[model] = {}
    best[model] = {}
    best[model]["config"] = 0
    best[model]["loss"] = 1000000
    model_instants[model] = {}

    if model.__name__ == "Conv1DNet":
        for i, config in enumerate(cnn_configs):
            logging[model][i] = {}
            model_instants[model][i] = model().arch(n_layers=config[0],
                                                    n_filters=config[1],
                                                    kernel_size=config[2],
                                                    dropout_rate=config[3])
    else:
        for i, config in enumerate(rnn_configs):
            logging[model][i] = {}
            model_instants[model][i] = model().arch(n_layers=config[0],
                                                    hidden_size=config[1],
                                                    dropout_rate=config[2])

# Tuning hyperparams
for model in models:
    loss_list = [[0]*n_folds]*len(model_instants[model])

    fold_idx = 0
    for train, validate in kfold.split(x_train_scaled, y_train_scaled):
        for i, (_, model_instant) in enumerate(model_instants[model].items()):
            model_instant.fit(x_train_scaled[train], y_train_scaled[train], batch_size=32, epochs=n_epochs)
            result = model_instant.evaluate(x_train_scaled[validate], y_train_scaled[validate], batch_size=32)
            loss_list[i][fold_idx] = result
        fold_idx += 1

    for i, _ in enumerate(model_instants[model]):
        logging[model][i]["mean_loss"] = np.mean(loss_list[i])
        logging[model][i]["std_loss"] = np.std(loss_list[i])
        if logging[model][i]["mean_loss"] < best[model]["loss"]:
            best[model]["loss"] = logging[model][i]["mean_loss"]
            best[model]["config"] = i

    mean_losses = []
    std_losses = []
    for _, i in logging[model].items():
        mean_losses.append(i["mean_loss"])
        std_losses.append(i["std_loss"])
    mean_losses, std_losses = np.array(mean_losses), np.array(std_losses)
    config_idx_range = np.arange(0, len(model_instants[model]), 1)

    plt.figure()
    plt.plot(config_idx_range, mean_losses, label="Validation Curve", color="black")
    plt.fill_between(config_idx_range, mean_losses - std_losses, mean_losses + std_losses, color="gray")
    plt.title("Validation Curve for " + model.__name__ + "Lowest loss acquired with config number " + str(best[model]["config"]))
    plt.xlabel("Config id")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.savefig('./figure/' + model.__name__ + '_validation curve.png', bbox_inches='tight')


