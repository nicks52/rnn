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


def split_input_data(input_data, train_split, memory_length):
    """Returns x_train, y_train, x_test, y_test numpy arrays

    The train and test arrays are separated by the value train_split.
    The value train_split specifies how many samples will be in the
    train_data set and the remaining samples will be in the test_data
    set.

    The train and test arrays are then split into x and y values
    """
    training_data, testing_data = np.split(input_data, [train_split])
    x_train, y_train = split_to_x_y_data(training_data, memory_length)
    x_test, y_test = split_to_x_y_data(testing_data, memory_length)
    return x_train, y_train, x_test, y_test


def split_to_x_y_data(given_array, memory_length):
    """Returns x and y data based on the given array

    The model should predict the next value of x given x_k which means
    that y_k = x_{k+1}. So the data must be divided to follow this
    requirement.

    The model will also add a padding of zeros before x and y equal to
    memory_length - 1. This is to allow for the model to learn from the
    entire memory length while training.
    """
    zero_pad_array = left_pad_array_with_zeros(given_array, memory_length)
    x_data = np.delete(zero_pad_array, -1, axis=1)
    y_data = np.delete(zero_pad_array, 0, axis=1)

    # Reshaping x_data to run through LSTM model
    x_data_reshape = np.reshape(x_data, (x_data.shape[0], 1, x_data.shape[1]))

    return x_data_reshape, y_data


def left_pad_array_with_zeros(initial_array, number_of_zeros):
    """Returns an array that has been left padded with zeros

    Each array in the 2D array input will be padded with zeros of length
    number_of_zeros.
    """
    zeros_array = np.zeros((initial_array.shape[0], number_of_zeros))
    new_array = np.concatenate((zeros_array, initial_array), axis=1)
    return new_array


def build_rnn_model(memory_length):
    """Returns a built and compiled rnn model"""
    samples_per_signal = 500 - 1 + memory_length
    input_signal_shape = (1, samples_per_signal)

    model = Sequential()
    model.add(
        LSTM(memory_length,
             input_shape=input_signal_shape,
             return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(memory_length, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(memory_length))
    model.add(Dense(514, activation=None))

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


def plot_comparison(hist1, hist2):
    """
    Generate two plots to compare two RNN models.

    The first plot compares the training mean squared error (MSE) of the
    two models. And the second plot compares the validation mean squared
    error.

    Arguments:
      hist1: History object, the history for model1
      hist2: History object, the history for model2

    Results:
      One plot of the training MSE and one plot of the validation MSE.
    """
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(hist1.history['mse'], label = 'Base Model train', color = 'orange')
    ax[0].plot(hist2.history['mse'], label = 'Updated Model train', color = 'r')
    ax[0].set_title('Training MSE for Base Model and Updated Model')
    ax[0].set_ylabel('MSE')
    ax[0].set_xlabel('epoch')
    ax[0].ticklabel_format(useOffset=False)
    ax[0].legend()

    ax[1].plot(hist1.history['val_mse'], label = 'Base Model validation', color = 'orange')
    ax[1].plot(hist2.history['val_mse'], label = 'Updated Model validation', color = 'r')
    ax[1].set_title('Validation MSE for Base Model and Updated Model')
    ax[1].set_ylabel('MSE')
    ax[1].set_xlabel('epoch')
    ax[1].ticklabel_format(useOffset=False)
    ax[1].legend()

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

    # Build and compile the model
    built_rnn_model = build_rnn_model(memory_length)
    print(built_rnn_model.summary())

    # Train the model
    batch_size = 20
    epochs = 15
    rnn_model_history = train_rnn_model(built_rnn_model, x_train, y_train,
                                        batch_size, epochs)

    # Generate plot comparison for two models
    plot_comparison(rnn_model_history, rnn_model_history)

    # # Test the model
    # test_score, test_accuracy = evaluate_rnn_model(built_rnn_model, x_test,
    #                                                y_test, batch_size)

    # print('Test score:', test_score)
    # print('Test accuracy:', test_accuracy)
