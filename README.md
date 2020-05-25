# Building a Deep RNN to Predict the Next Signal in Time Series Data
### By Nick Serger

This project builds several deep RNNs to predict the next signal of time series data. The data file “data.npy” contains a matrix, of 100 rows and 500 columns. Each row represents a signal, which contains the superposition of a large number (larger than 20) of sinusoids with randomly generated amplitudes, frequencies, and phases.

To evaluate different features of deep RNN models, you can edit the evaluate_rnns.py file. No other files need to be edited to compare different RNNs.

The Deep RNN model that I found worked best has a memory length of 20, uses GRU RNN layers, uses hyperbolic tangent activation functions, uses the Nadam optimizer, is trained with batch sizes of 3, and is trained for 40 epochs.
