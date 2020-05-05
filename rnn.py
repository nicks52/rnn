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

def split_input_data(input_data):
    """Returns train_data and test_data numpy arrays with 66/34 split"""
    train_data, test_data = np.split(input_data, [66])
    return train_data, test_data


# TODO: Write a build model function
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
    train_data, test_data = split_input_data(input_data)
