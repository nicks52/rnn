#!/usr/bin/env python
"""
By Nick Serger

A Deep RNN to predict future timeseries data.

The data file “data.npy” contains a matrix, of 100 rows and 500 columns.
Each row represents a signal, which contains the superposition of a
large number (larger than 20) of sinusoids with randomly generated
amplitudes, frequencies, and phases.


TODO:
Make a Preprocessing Class
Make a rnn model class
Make an evaluation scipt

"""

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM, GRU, SimpleRNN

from sklearn.preprocessing import StandardScaler

# Increase font size of matplotlib fonts
plt.rcParams.update({'font.size': 12})


def load_data(data_filename):
    """Returns the input data from the file data.npy as a numpy array"""
    try:
        loaded_data = np.load(data_filename)
    except FileNotFoundError as err:
        print('data.npy file was not found at path {}'.format(data_filename))
        raise err
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(loaded_data)
    return standardized_data


def split_input_data(input_data, train_split):
    """Returns x_train, y_train, x_test, y_test numpy arrays

    The train and test arrays are separated by the value train_split.
    The value train_split specifies how many samples will be in the
    train_data set and the remaining samples will be in the test_data
    set.

    The train and test arrays are then split into x and y values
    """
    training_data, testing_data = np.split(input_data, [train_split])
    x_train, y_train = split_to_x_y_data(training_data)
    x_test, y_test = split_to_x_y_data(testing_data)
    return x_train, y_train, x_test, y_test


def split_to_x_y_data(given_array):
    """Returns x and y data based on the given array

    The model should predict the next value of x given x_k which means
    that y_k = x_{k+1}. So the data must be divided to follow this
    requirement.

    The model will also add a padding of zeros before x and y equal to
    memory_length - 1. This is to allow for the model to learn from the
    entire memory length while training.
    """
    # Reshape array to size (100, 500, 1), (66, 500, 1), or (34, 500, 1)
    reshaped_array = np.expand_dims(given_array, axis=2)
    x_data = reshaped_array[:, :-1, :]
    y_data = reshaped_array[:, 1:, :]
    return x_data, y_data


def build_rnn_model(memory_length,
                    rnn_layer='LSTM',
                    dropout=0.0,
                    activation='tanh',
                    batch_norm=True,
                    optimizer='adam'):
    """Returns a built and compiled rnn model"""

    if rnn_layer == 'LSTM':
        model = Sequential()
        model.add(
            LSTM(memory_length,
                 input_shape=(499, 1),
                 activation=activation,
                 dropout=dropout,
                 recurrent_dropout=dropout,
                 return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            LSTM(memory_length,
                 activation=activation,
                 dropout=dropout,
                 recurrent_dropout=dropout,
                 return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            LSTM(memory_length,
                 activation=activation,
                 dropout=dropout,
                 recurrent_dropout=dropout,
                 return_sequences=True))
        model.add(Dense(1, activation=None))
    elif rnn_layer == 'GRU':
        model = Sequential()
        model.add(
            GRU(memory_length,
                input_shape=(499, 1),
                activation=activation,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            GRU(memory_length,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            GRU(memory_length,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=True))
        model.add(Dense(1, activation=None))
    else:
        model = Sequential()
        model.add(
            SimpleRNN(memory_length,
                      input_shape=(499, 1),
                      activation=activation,
                      dropout=dropout,
                      recurrent_dropout=dropout,
                      return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            SimpleRNN(memory_length,
                      activation=activation,
                      dropout=dropout,
                      recurrent_dropout=dropout,
                      return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            SimpleRNN(memory_length,
                      activation=activation,
                      dropout=dropout,
                      recurrent_dropout=dropout,
                      return_sequences=True))
        model.add(Dense(1, activation=None))

    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])

    return model


def train_rnn_model(compile_model, x_train, y_train, batch_size, epochs):
    """Returns model trained on x_train and y_train data"""
    train_model = compile_model.fit(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_split=0.2)

    return train_model


def evaluate_rnn_model(train_model, x_test, y_test, batch_size):
    """Returns score and MSE of the trained RNN model"""
    score, mse = train_model.evaluate(x_test, y_test, batch_size=batch_size)

    return score, mse


def plot_comparison(histories, names, title):
    """
    Generate two plots to compare RNN models.

    The first plot compares the training mean squared error (MSE) of the
    models. And the second plot compares the validation mean squared
    error.

    Args:
      histories: list of history objects from each time a new model is
        trained
      names: list of strings to uniquely identify the history objects
      title: string, super title of the figure
    """
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)

    for index, history in enumerate(histories):
        if not names:
            ax[0].plot(history.history['mse'])
        else:
            ax[0].plot(history.history['mse'], label=names[index])
            ax[0].legend()
        ax[0].set_title('Training MSE')
        ax[0].set_ylabel('MSE')
        ax[0].set_xlabel('epoch')
        ax[0].ticklabel_format(useOffset=False)

        if not names:  # If names is empty
            ax[1].plot(history.history['val_mse'])
        else:
            ax[1].plot(history.history['val_mse'], label=names[index])
            ax[1].legend()
        ax[1].set_title('Validation MSE')
        ax[1].set_ylabel('Validation MSE')
        ax[1].set_xlabel('epoch')
        ax[1].ticklabel_format(useOffset=False)

    plt.show()


if __name__ == '__main__':
    #Input the data
    path_to_data_file = 'data.npy'
    all_data = load_data(path_to_data_file)

    #Separate into train and test data
    train_test_split = 66
    x_train, y_train, x_test, y_test = split_input_data(
        all_data, train_test_split)

    # # Evaluate varying batch_size on model performance
    # histories = []
    # names = []
    # epochs = 100
    # memory_length = 15

    # batch_sizes = [30, 20, 10, 5, 1]
    # for batch in batch_sizes:
    #     built_rnn_model = build_rnn_model(memory_length)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch, epochs)
    #     histories.append(rnn_model_history)
    #     names.append('Batch size: {}'.format(batch))

    # title = 'Evaluation of Batch Size on Model Performance'
    # plot_comparison(histories, names, title)

    # # Evaluating number of epochs used in training on model performance
    # batch_size = 5
    # epochs = 1000
    # memory_length = 15

    # built_rnn_model = build_rnn_model(memory_length)
    # rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                     batch_size, epochs)
    # plot_comparison([rnn_model_history], [], 'Evaluation of Training for 1,000 Epochs')

    # # Test RNN layers
    # histories = []
    # names = []
    # epochs = 100
    # batch_size = 5
    # memory_length = 15

    # rnn_layers = ['LSTM', 'GRU', 'SimpleRNN']
    # for rnn_layer in rnn_layers:
    #     built_rnn_model = build_rnn_model(memory_length, rnn_layer)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch_size, epochs)
    #     histories.append(rnn_model_history)
    #     names.append(rnn_layer)

    # title = 'Evaluation of Different RNN Layers on Model Performance'
    # plot_comparison(histories, names, title)

    # # Test RNN layers with and without dropout
    # histories = []
    # names = []
    # epochs = 100
    # batch_size = 5
    # memory_length = 15
    # title = 'Evaluation of Different RNN Layers With and Without Dropout on Model Performance'

    # rnn_layers = ['LSTM', 'GRU', 'SimpleRNN']
    # dropout_values = [0.0, 0.2]
    # for rnn_layer in rnn_layers:
    #     for dropout in dropout_values:
    #         built_rnn_model = build_rnn_model(memory_length, rnn_layer,
    #                                           dropout)
    #         rnn_model_history = train_rnn_model(built_rnn_model, x_train,
    #                                             y_train, batch_size, epochs)
    #         histories.append(rnn_model_history)
    #         names.append('{}, dropout: {}'.format(rnn_layer, dropout))

    # title = 'Evaluation of Different RNN Layers With and Without Dropout on Model Performance'
    # plot_comparison(histories, names, title)

    # # Test memory_length
    # histories = []
    # names = []
    # epochs = 100
    # batch_size = 5

    # memory_lengths = [5, 10, 15]
    # for memory_length in memory_lengths:
    #     built_rnn_model = build_rnn_model(memory_length, 'GRU', dropout=0.0)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch_size, epochs)
    #     histories.append(rnn_model_history)
    #     names.append('Memory Length: {}'.format(memory_length))

    # title = 'Evaluation of Memory Length on Model Performance'
    # plot_comparison(histories, names, title)

    # Test different activation functions
    histories = []
    names = []
    epochs = 100
    batch_size = 5
    memory_length = 15
    rnn_layer = 'GRU'
    dropout = 0.0

    activation_functions = ['tanh', 'relu', 'sigmoid']
    for activation_function in activation_functions:
        built_rnn_model = build_rnn_model(memory_length, rnn_layer, dropout,
                                          activation_function)
        rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
                                            batch_size, epochs)
        histories.append(rnn_model_history)
        names.append('{}'.format(activation_function))

    title = 'Evaluation of Different Activation Functions on Model Performance'
    plot_comparison(histories, names, title)

    # # Test using and not using batch normalization
    # histories = []
    # names = []
    # epochs = 100
    # batch_size = 5
    # memory_length = 15
    # rnn_layer = 'GRU'
    # dropout = 0.0
    # activation_function = 'tanh'

    # use_batch_norm = [True, False]
    # for batch_norm in use_batch_norm:
    #     built_rnn_model = build_rnn_model(memory_length, rnn_layer, dropout,
    #                                       activation_function, batch_norm)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch_size, epochs)
    #     histories.append(rnn_model_history)
    # names = ['With Batch Normalization', 'Without Batch Normalization']

    # title = 'Evaluation of Batch Normalization on Model Performance'
    # plot_comparison(histories, names, title)

    # # Test different optimizers
    # histories = []
    # names = []
    # epochs = 100
    # batch_size = 5
    # memory_length = 15
    # rnn_layer = 'GRU'
    # dropout = 0.0
    # activation_function = 'tanh'
    # batch_norm = False

    # optimizers = ['adam', 'sgd', 'rmsprop', 'adadelta', 'nadam']
    # for optimizer in optimizers:
    #     built_rnn_model = build_rnn_model(memory_length, rnn_layer, dropout,
    #                                       activation_function, batch_norm,
    #                                       optimizer)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch_size, epochs)
    #     histories.append(rnn_model_history)
    #     names.append('{}'.format(optimizer))

    # title = 'Evaluation of Different Optimizers on Model Performance'
    # plot_comparison(histories, names, title)

    # # Final Model
    # epochs = 40
    # batch_size = 5
    # memory_length = 15
    # rnn_layer = 'GRU'
    # dropout = 0.0
    # activation_function = 'tanh'
    # batch_norm = False
    # optimizer = 'adam'
    # built_rnn_model = build_rnn_model(memory_length, rnn_layer, dropout,
    #                                   activation_function, batch_norm,
    #                                   optimizer)
    # rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                     batch_size, epochs)

    # plot_comparison([rnn_model_history], ['Final Model'],
    #                 'Final Model Performance')

    # # Test the model
    # _, test_mse = evaluate_rnn_model(built_rnn_model, x_test, y_test,
    #                                           batch_size)

    # print('Test MSE:', test_mse)
