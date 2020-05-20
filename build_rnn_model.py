#!/usr/bin/env python
"""
By Nick Serger

A class to build a train RNN models using keras.
"""

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM, GRU, SimpleRNN

import load_rnn_data


class RNN:
    def __init__(self,
                 memory_length,
                 input_shape,
                 rnn_layer='LSTM',
                 activation='tanh',
                 batch_norm=False,
                 optimizer='adam'):
        """Initializes RNN class."""
        self.memory_length = memory_length
        self.input_shape = input_shape
        self.rnn_layer = rnn_layer
        self.activation = activation
        self.batch_norm = batch_norm
        self.optimizer = optimizer

        self.model = self._build_rnn_model()

    def _build_rnn_model(self):
        """Returns a built and compiled rnn model"""
        if rnn_layer == 'LSTM':
            self.model = self._build_lstm_model()
        elif rnn_layer == 'GRU':
            self.model = self._build_gru_model()
        else:
            self.model = self._build_simple_rnn_model()

        self.model.compile(optimizer=self.optimizer,
                           loss="mse",
                           metrics=["mse"])

        return self.model

    def _build_lstm_model(self):
        """Returns a lstm model"""
        model = Sequential()
        model.add(
            LSTM(self.memory_length,
                 input_shape=self.input_shape,
                 activation=self.activation,
                 return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            LSTM(self.memory_length,
                 activation=self.activation,
                 return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            LSTM(self.memory_length,
                 activation=self.activation,
                 return_sequences=True))
        model.add(Dense(1, activation=None))

        return model

    def _build_simple_rnn_model(self):
        """Returns a simple rnn model"""
        model = Sequential()
        model.add(
            SimpleRNN(memory_length,
                      input_shape=self.input_shape,
                      activation=self.activation,
                      return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            SimpleRNN(self.memory_length,
                      activation=self.activation,
                      return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            SimpleRNN(self.memory_length,
                      activation=self.activation,
                      return_sequences=True))
        model.add(Dense(1, activation=None))

        return model

    def _build_gru_model(self):
        """Returns a gru model"""
        model = Sequential()
        model.add(
            GRU(self.memory_length,
                input_shape=self.input_shape,
                activation=self.activation,
                return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            GRU(self.memory_length,
                activation=self.activation,
                return_sequences=True))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(
            GRU(self.memory_length,
                activation=self.activation,
                return_sequences=True))
        model.add(Dense(1, activation=None))

        return model

    def train_rnn_model(self, x_train, y_train, batch_size, epochs):
        """Returns model history trained on x_train and y_train data"""
        model_history = self.model.fit(x_train,
                                       y_train,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       validation_split=0.2)

        return model_history

    def evaluate_rnn_model(self, x_test, y_test, batch_size):
        """Returns MSE of the trained RNN model"""
        _, mse = self.model.evaluate(x_test, y_test, batch_size=batch_size)

        return mse


if __name__ == '__main__':
    (x_train, y_train), (x_test,
                         y_test) = load_rnn_data.load_data('data.npy',
                                                           normalize_data=True,
                                                           train_split=.66)
