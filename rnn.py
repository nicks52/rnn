#!/usr/bin/env python
#By Nick Serger
"""
The data file “data.npy” contains a matrix, of 100 rows and 500 columns.
Each row represents a signal, which contains the superposition of a
large number (larger than 20) of sinusoids with randomly generated
amplitudes, frequencies, and phases.
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

from sklearn.preprocessing import StandardScaler


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

    reshaped_array = np.expand_dims(given_array, axis=2)
    x_data = reshaped_array[:, :-1, :]
    y_data = reshaped_array[:, 1:, :]
    return x_data, y_data


def build_rnn_model(memory_length):
    """Returns a built and compiled rnn model"""
    model = Sequential()
    model.add(LSTM(memory_length, input_shape=(499, 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(memory_length, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(memory_length, return_sequences=True))
    model.add(Dense(1, activation=None))

    model.compile(optimizer="Adam", loss="mse", metrics=["mse"])

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
    """Returns score and accuracy of the trained RNN model"""
    score, accuracy = train_model.evaluate(x_test,
                                           y_test,
                                           batch_size=batch_size)

    return score, accuracy


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


        if not names: # If names is empty
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
    memory_length = 15
    x_train, y_train, x_test, y_test = split_input_data(
        all_data, train_test_split, memory_length)

    # # Evaluate varying batch_size on model performance
    # histories = []
    # names = []
    # epochs = 100

    # batch_sizes = [30, 20, 10, 5, 1]
    # for batch in batch_sizes:
    #     built_rnn_model = build_rnn_model(memory_length)
    #     rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
    #                                         batch, epochs)
    #     histories.append(rnn_model_history)
    #     names.append('Batch size: {}'.format(batch))

    # title = 'Evaluation of Batch Size on Model Performance'

    # # Generate plot comparison for two models
    # plot_comparison(histories, names, title)

    # Using batch size of 5 for remainder of analysis
    # Evaluating number of epochs used in training on model performance
    batch_size = 5
    epochs = 1000
    histories = []
    names = []

    built_rnn_model = build_rnn_model(memory_length)
    rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
                                        batch_size, epochs)
    histories.append(rnn_model_history)
    names.append()

    # # Test the model
    # test_score, test_accuracy = evaluate_rnn_model(built_rnn_model, x_test,
    #                                                y_test, batch_size)

    # print('Test score:', test_score)
    # print('Test accuracy:', test_accuracy)
