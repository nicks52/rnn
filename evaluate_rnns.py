#!/usr/bin/env python
"""
By Nick Serger

A module to evaluate rnn models.
"""

import load_rnn_data
import rnn
import compare_rnn_models

# Evaluating different features of RNNs to determine the optimal RNN design

# batch_size_histories, batch_size_names = compare_rnn_models.compare_feature(
#     'batch_size', [3, 5, 10, 20])
# batch_size_title = 'Different Batch Size Values'
# compare_rnn_models.plot_comparison(batch_size_histories, batch_size_names,
#                                    batch_size_title)

# memory_histories, memory_names = compare_rnn_models.compare_feature(
#     'memory_length', [1, 5, 10, 15, 20])
# memory_title = 'Different Memory Length Values'
# compare_rnn_models.plot_comparison(memory_histories, memory_names,
#                                    memory_title)

# rnn_layer_histories, rnn_layer_names = compare_rnn_models.compare_feature(
#     'rnn_layer', ['LSTM', 'GRU', 'SimpleRNN'])
# rnn_layer_title = 'Different RNN Layers'
# compare_rnn_models.plot_comparison(rnn_layer_histories, rnn_layer_names,
#                                    rnn_layer_title)

# activation_histories, activation_names = compare_rnn_models.compare_feature(
#     'activation', ['tanh', 'relu', 'sigmoid'])
# activation_title = 'Different Activation Functions'
# compare_rnn_models.plot_comparison(activation_histories, activation_names,
#                                    activation_title)

# batch_norm_histories, batch_norm_names = compare_rnn_models.compare_feature(
#     'batch_norm', [True, False])
# batch_norm_title = 'Batch Normalization'
# compare_rnn_models.plot_comparison(batch_norm_histories, batch_norm_names,
#                                    batch_norm_title)

# optimizer_histories, optimizer_names = compare_rnn_models.compare_feature(
#     'optimizer', ['adam', 'sgd', 'rmsprop', 'adadelta', 'nadam'])
# optimizer_title = 'Different Optimizers'
# compare_rnn_models.plot_comparison(optimizer_histories, optimizer_names,
#                                    optimizer_title)

# Based on the above comparisons, an optimal RNN would have
epochs = 40
batch_size = 3

memory_length = 20
input_shape = (499, 1)
rnn_layer = 'GRU'
activation = 'tanh'
batch_norm = False
optimizer = 'nadam'

# Load the training and testing data
(x_train, y_train), (x_test,
                     y_test) = load_rnn_data.load_data('data.npy',
                                                       normalize_data=True,
                                                       train_split=.66)

# Build and train the model
test_rnn = rnn.RNN(input_shape, memory_length, rnn_layer, activation,
                   batch_norm, optimizer)

rnn_model_history = test_rnn.train_rnn_model(x_train, y_train, batch_size,
                                             epochs)

# Evaluate the final model
test_mse = test_rnn.evaluate_rnn_model(x_test, y_test, batch_size)

print('Test MSE: {}'.format(test_mse))
