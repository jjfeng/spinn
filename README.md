# Sparse-Input Neural Network

## Installation:
Install Tensorflow (python).

## Running:
To perform classification on a dataset, use `spinn_classification.py`.
For regression on a dataset, use `spinn_regression.py`.
There will be two output files: one is the log file and one is the fitted models.
The fitted models will be a list of all the fitted models throughout the training process.
The last one in the list is the final fitted model.
The fitted models are represented by the `NeuralNetworkResult` class.

For an example for how to perform simulations on the data, look at `spinn_simulation.py`.
There will be three output files: the log file, the fitted models, and the simulated data.
