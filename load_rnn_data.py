#!/usr/bin/env python
"""
By Nick Serger

A function to handle preprocessing data to run through a keras built RNN.

The data file “data.npy” contains a matrix, of 100 rows and 500 columns.
Each row represents a signal, which contains the superposition of a
large number (larger than 20) of sinusoids with randomly generated
amplitudes, frequencies, and phases.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(file_path, normalize_data=True, train_split=.66):
    """Load the data in the path provided.

    Arguments:
      file_path: string, path to file to import
      scale_data: boolean, whether the data should be scaled.
      train_split: float between 0 and 1 that is the percent of the
        data to be split as training data, the remaining data will
        be used as test data.

    Returns:
      Tuple of numpy arrays: (x_train, y_train), (x_test, y_test)
    """
    input_data = _load_data(file_path)
    if normalize_data:
        input_data = _scale_data(input_data)
    return _split_input_data(input_data, train_split)


def _load_data(file_path):
    """Returns the data from the input file as a numpy array."""
    try:
        loaded_data = np.load(file_path)
    except FileNotFoundError as err:
        print('data file was not found at path {}'.format(file_path))
        raise err
    return loaded_data


def _scale_data(loaded_data):
    """Normalize data with a mean of 0 and std dev of 1"""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(loaded_data)
    return standardized_data


def _split_input_data(input_data, train_split):
    """Returns x_train, y_train, x_test, y_test numpy arrays.

    The train and test arrays are separated by the value train_split.
    The value train_split specifies the percent of how many samples
    will be in the train_data set and the remaining samples will be
    in the test_data set.

    The train and test arrays are then split into x and y values
    """
    train_samples = int(input_data.shape[0] * train_split)

    training_data, testing_data = np.split(input_data, [train_samples])
    x_train, y_train = _split_to_x_y_data(training_data)
    x_test, y_test = _split_to_x_y_data(testing_data)
    return (x_train, y_train), (x_test, y_test)


def _split_to_x_y_data(given_array):
    """Returns x and y data based on the given array.

    The model should predict the next value of x given x_k which means
    that y_k = x_{k+1}. So the data must be divided to follow this
    requirement.

    The model will also add a padding of zeros before x and y equal to
    memory_length - 1. This is to allow for the model to learn from the
    entire memory length while training.
    """
    # Reshape array to from size (x, y) to (x, y, 1)
    reshaped_array = np.expand_dims(given_array, axis=2)
    x_data = reshaped_array[:, :-1, :]
    y_data = reshaped_array[:, 1:, :]
    return x_data, y_data
