#!/usr/bin/env python
#By Nick Serger
"""
The data file “data.npy” contains a matrix, of 100 rows and 500 columns.
Each row represents a signal, which contains the superposition of a
large number (larger than 20) of sinusoids with randomly generated
amplitudes, frequencies, and phases.

# Project outline:
    import functions

    load data
    divide data into train and test data

    build model
    compile model

    train model
    validate model

    predict next value

    test model
"""

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization


def load_data(data_filename):
    """Returns the input data from the file data.npy as a numpy array"""
    try:
        loaded_data = np.load(data_filename)
    except FileNotFoundError as err:
        print('data.npy file was not found at path {}'.format(data_filename))
        raise err
    return loaded_data


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
    return x_data, y_data


def left_pad_array_with_zeros(initial_array, number_of_zeros):
    """Returns an array that has been left padded with zeros

    Each array in the 2D array input will be padded with zeros of length
    number_of_zeros.
    """
    zeros_array = np.zeros((initial_array.shape[0], number_of_zeros))
    new_array = np.concatenate((zeros_array, initial_array), axis=1)
    return new_array


def build_rnn_model(memory_length, signal_length):
    """Returns a built and compiled rnn model"""
    model = Sequential()
    model.add(
        LSTM(memory_length,
             input_shape=(1, signal_length),
             dropout=0.2,
             recurrent_dropout=0.2,
             return_sequences=True))
    model.add(BatchNormalization())
    model.add(
        LSTM(memory_length, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(memory_length, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation=None))

    model.compile(optimizer="Adam", loss="mse", metrics=["mse"])

    return model


def train_rnn_model(compile_model, x_train, y_train, batch_size):
    """Returns model trained on x_train and y_train data"""
    train_model = compile_model.fit(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    epochs=15,
                                    validation_split=0.2)

    return train_model


def evaluate_rnn_model(train_model, x_test, y_test, batch_size):
    """Returns score and accuracy of the trained RNN model"""
    score, accuracy = train_model.evaluate(x_test,
                                           y_test,
                                           batch_size=batch_size)

    return score, accuracy


if __name__ == '__main__':
    #Input the data
    path_to_data_file = 'data.npy'
    all_data = load_data(path_to_data_file)

    #Separate into train and test data
    train_test_split = 66
    memory_length = 15
    x_train, y_train, x_test, y_test = split_input_data(all_data, train_test_split, memory_length)

    # Build and compile the model
    built_rnn_model = build_rnn_model(memory_length, x_train.shape[1])

    # Train the model

    #TODO: reshape the array to have shape (None, 66, 514) from (66, 514)
    #TODO: run yapf and pylint

    batch_size = 20
    trained_rnn_model = train_rnn_model(built_rnn_model, x_train, y_train, batch_size)

    # # Test the model
    # test_score, test_accuracy = evaluate_rnn_model(trained_rnn_model, x_test,
    #                                                y_test, batch_size)

    # print('Test score:', test_score)
    # print('Test accuracy:', test_accuracy)
