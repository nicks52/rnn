#!/usr/bin/env python
"""
By Nick Serger

A module with functions to build rnn models and compare rnn models.
"""

import matplotlib.pyplot as plt

import load_rnn_data
import rnn

# Increase font size of matplotlib fonts
plt.rcParams.update({'font.size': 12})


def plot_comparison(histories, names, title):
    """
    Generate two plots to compare RNN models.

    The first plot compares the training mean squared error (MSE) of the
    models. And the second plot compares the validation mean squared
    error.

    Arguments:
        histories: list of history objects from each time a new model is
            trained.
        names: list of strings to uniquely identify the history objects,
            these are included in the legend of the plot.
        title: string, super title of the figure.
    """
    fig, ax = plt.subplots(1, 2)
    fig_title = 'Evaluation of {} on Model Performance'.format(title)
    fig.suptitle(fig_title)

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


def compare_feature(feature_name,
                    feature_inputs,
                    data_filepath='data.npy',
                    normalize_data=True,
                    train_split=.66,
                    epochs=40,
                    batch_size=5,
                    input_shape=(499, 1),
                    memory_length=15,
                    rnn_layer='LSTM',
                    activation='tanh',
                    batch_norm=False,
                    optimizer='adam'):
    """Build and train a RNN model for each feature input.

    Given a feature_name, RNN models will be built that vary by the
    feature_input value. For example, if feature_name='batch_size' and
    feature_inputs=[1,5,10,50], then 4 RNN models will be built and each
    will be trained with a batch size of 1, 5, 10, or 50. All additional
    RNN parameters have default values that can be changed. If the
    parameter value is changed, it will be changed for all models. If
    the parameter for the feature_name is also specified, the value in
    the parameter will be ignored and the different feature_inputs
    values will be used.

    Arguments:
        feature_name: name of the feature to be evaluated. Options are:
            epochs, batch_size, memory_length, rnn_layer, activation,
            batch_norm, optimizer.
        feature_inputs: list of possible feature inputs to iterate over.
            I.e. if feature_name=batch_size, feature_inputs=[1,5,10,50].
        data_filepath: string, path to file to import
        normalize_data: boolean, whether the data should be scaled.
        train_split: float between 0 and 1 that is the percent of the
            data to be split as training data, the remaining data will
            be used as test data.
        epochs: int, number of epochs to train the model.
        batch_size: int, number of samples per batch used in training.
        input_shape: tuple of ints, the first value is the number of
            time samples and the second value is the number of features.
        memory_length: int, number of units remembered during training.
        rnn_layer: string, rnn_layer used to build the model. Either 
            'LSTM', 'GRU', 'SimpleRNN'.
        activation: string, activation layer to use in the RNN. Must be
            a valid keras activation layer.
        batch_norm: boolean, set to True to use BatchNormalization
            layers.
        optimizer: string, optimizer used to train the RNN. Must be a
            valid keras optimizer.

    Returns:
        list of histories and list of names that correspond to each
            feature_inputs value.
    """

    (x_train, y_train), (x_test, y_test) = load_rnn_data.load_data(
        data_filepath, normalize_data, train_split)

    if feature_name == 'batch_size':
        histories, names = _compare_batch_size(feature_inputs, x_train,
                                               y_train, epochs, input_shape,
                                               memory_length, rnn_layer,
                                               activation, batch_norm,
                                               optimizer)
    elif feature_name == 'memory_length':
        histories, names = _compare_memory_length(feature_inputs, x_train,
                                                  y_train, epochs, batch_size,
                                                  input_shape, rnn_layer,
                                                  activation, batch_norm,
                                                  optimizer)
    elif feature_name == 'rnn_layer':
        histories, names = _compare_rnn_layer(feature_inputs, x_train, y_train,
                                              epochs, batch_size, input_shape,
                                              memory_length, activation,
                                              batch_norm, optimizer)
    elif feature_name == 'activation':
        histories, names = _compare_activation(feature_inputs, x_train,
                                               y_train, epochs, batch_size,
                                               input_shape, memory_length,
                                               rnn_layer, batch_norm,
                                               optimizer)
    elif feature_name == 'batch_norm':
        histories, names = _compare_batch_norm(feature_inputs, x_train,
                                               y_train, epochs, batch_size,
                                               input_shape, memory_length,
                                               rnn_layer, activation,
                                               optimizer)
    elif feature_name == 'optimizer':
        histories, names = _compare_optimizer(feature_inputs, x_train, y_train,
                                              epochs, batch_size, input_shape,
                                              memory_length, rnn_layer,
                                              activation, batch_norm)
    else:
        print(
            'Please only provide feature_name values in the list [batch_size,'
            ' memory_length, rnn_layer, activation, batch_norm, optimizer]')
        return None

    return histories, names


def _compare_batch_size(batch_sizes,
                        x_train,
                        y_train,
                        epochs=40,
                        input_shape=(499, 1),
                        memory_length=15,
                        rnn_layer='LSTM',
                        activation='tanh',
                        batch_norm=False,
                        optimizer='adam'):
    """Builds and trains models with varying batch size values."""
    histories = []
    names = []

    for batch in batch_sizes:
        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train, batch,
                                                      epochs)

        histories.append(rnn_model_history)
        names.append('Batch size: {}'.format(batch))
    return histories, names


def _compare_memory_length(memory_lengths,
                           x_train,
                           y_train,
                           epochs=40,
                           batch_size=5,
                           input_shape=(499, 1),
                           rnn_layer='LSTM',
                           activation='tanh',
                           batch_norm=False,
                           optimizer='adam'):
    """Builds and trains models with varying memory length values."""
    histories = []
    names = []

    for memory_length in memory_lengths:
        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train,
                                                      batch_size, epochs)

        histories.append(rnn_model_history)
        names.append('Memory Length: {}'.format(memory_length))
    return histories, names


def _compare_rnn_layer(rnn_layers,
                       x_train,
                       y_train,
                       epochs=40,
                       batch_size=5,
                       input_shape=(499, 1),
                       memory_length=15,
                       activation='tanh',
                       batch_norm=False,
                       optimizer='adam'):
    """Builds and trains models with different types of RNN layers."""
    histories = []
    names = []

    for rnn_layer in rnn_layers:
        if rnn_layer.lower() not in ['lstm', 'gru', 'simplernn']:
            print('{} is not a valid RNN layer. Only LSTM, GRU, or SimpleRNN'
                  ' RNN layers are acceptable.'.format(rnn_layer))
            raise ValueError

        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train,
                                                      batch_size, epochs)

        histories.append(rnn_model_history)
        names.append(rnn_layer)
    return histories, names


def _compare_activation(activations,
                        x_train,
                        y_train,
                        epochs=40,
                        batch_size=5,
                        input_shape=(499, 1),
                        memory_length=15,
                        rnn_layer='LSTM',
                        batch_norm=False,
                        optimizer='adam'):
    """Builds and trains models with varying activation values."""
    histories = []
    names = []

    for activation in activations:
        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train,
                                                      batch_size, epochs)

        histories.append(rnn_model_history)
        names.append('{}'.format(activation))
    return histories, names


def _compare_batch_norm(batch_norms,
                        x_train,
                        y_train,
                        epochs=40,
                        batch_size=5,
                        input_shape=(499, 1),
                        memory_length=15,
                        rnn_layer='LSTM',
                        activation='tanh',
                        optimizer='adam'):
    """Builds and trains models with and without Batch Normalization."""
    histories = []
    names = []

    for batch_norm in batch_norms:
        if not isinstance(batch_norm, bool):
            print(
                '{} is not a boolean value. Only input boolean values'.format(
                    batch_norm))
            raise ValueError

        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train,
                                                      batch_size, epochs)

        histories.append(rnn_model_history)
        if batch_norm:
            names.append('With Batch Normalization')
        else:
            names.append('Without Batch Normalization')
    return histories, names


def _compare_optimizer(optimizers,
                       x_train,
                       y_train,
                       epochs=40,
                       batch_size=5,
                       input_shape=(499, 1),
                       memory_length=15,
                       rnn_layer='LSTM',
                       activation='tanh',
                       batch_norm=False):
    """Builds and trains models with varying optimizer values."""
    histories = []
    names = []

    for optimizer in optimizers:
        rnn_model = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                            batch_norm, optimizer)

        rnn_model_history = rnn_model.train_rnn_model(x_train, y_train,
                                                      batch_size, epochs)

        histories.append(rnn_model_history)
        names.append('{}'.format(optimizer))
    return histories, names
