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
from keras.models import Sequential
from keras.layers import Dense, Embedding, BatchNormalization
from keras.layers import LSTM

def load_data(data_path):
    """Returns the input data from the file data.npy as a numpy array"""
    try:
        input_data = np.load(data_path)
    except FileNotFoundError as err:
        print('data.npy file was not found at path {}'.format(data_path))
        raise err
    return input_data

def split_input_data(input_data, split):
    """Returns train_data and test_data numpy arrays with 66/34 split"""
    train_data, test_data = np.split(input_data, [split])
    return train_data, test_data

def build_rnn_model():
    """Returns an rnn model that still needs to be compiled"""
    model = Sequential()
    # model.add(Embedding(max_features, 100))
    model.add(LSTM(128, input_shape=(None, 500), 
        dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation=None))

    return model

# TODO: Write a compile model function

# TODO: Write a train model function
# TODO: Write a validate model function

# TODO: Predict the next value given the input

# TODO: Write an evaluate model function

if __name__ == '__main__':
    #Input the data
    data_path = 'data.npy'
    input_data = load_data(data_path)

    #Separate into train and test data
    split = 66
    train_data, test_data = split_input_data(input_data, split)

    print('Build model...')
    model = Sequential()
    # model.add(Embedding(max_features, 100))
    model.add(LSTM(128, input_shape=(None, 40, 40, 1), 
        padding='same', dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='sigmoid'))
