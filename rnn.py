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
from keras.layers import Dense, Embedding, BatchNormalization
from keras.layers import LSTM


def load_data(data_filename):
    """Returns the input data from the file data.npy as a numpy array"""
    try:
        loaded_data = np.load(data_filename)
    except FileNotFoundError as err:
        print('data.npy file was not found at path {}'.format(data_filename))
        raise err
    return loaded_data


def split_input_data(all_data, train_split):
    """Returns train_data and test_data numpy arrays

    The train and test arrays are separated by the value train_split.
    The value train_split specifies how many samples will be in the
    train_data set and the remaining samples will be in the test_data
    set.
    """
    training_data, testing_data = np.split(all_data, [train_split])
    return training_data, testing_data


def build_rnn_model():
    """Returns an rnn model that still needs to be compiled"""
    model = Sequential()
    model.add(
        LSTM(64,
             input_shape=(None, 1, 500),
             dropout=0.2,
             recurrent_dropout=0.2,
             return_sequences=True))
    model.add(BatchNormalization())
    model.add(
        LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(
        LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation=None))

    return model


def compile_rnn_model(built_model):
    """Returns the compiled model"""
    compile_model = built_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer='adam',
        metrics=[tf.keras.metrics.MeanSquaredError()])
    return compile_model


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
    input_data = load_data(path_to_data_file)

    #Separate into train and test data
    train_test_split = 66
    train_data, test_data = split_input_data(input_data, train_test_split)

    # Build and compile the model
    built_rnn_model = build_rnn_model()
    compiled_rnn_model = compile_rnn_model(built_rnn_model)

    # # Train the model
    # batch_size = 1
    # trained_rnn_model = train_rnn_model(compiled_rnn_model, x_train, y_train)

    # # Test the model
    # test_score, test_accuracy = evaluate_rnn_model(trained_rnn_model, x_test,
    #                                                y_test, batch_size)

    # print('Test score:', test_score)
    # print('Test accuracy:', test_accuracy)
