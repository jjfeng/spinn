# Sparse-Input Neural Network

Jean Feng and Noah Simon. "Sparse-Input Neural Networks for High-dimensional Nonparametric Regression and Classification." https://arxiv.org/abs/1711.07592

Note: For the latest iteration of this work, see Jean Feng and Noah Simon. "Ensembled sparse-input hierarchical networks for high-dimensional datasets." https://arxiv.org/abs/2005.04834

## Installation:
Install Tensorflow (python).

Code was run using python 3.4.3

## Primary files:
* `read_data.py`: make a dataset in the correct format
* `fit_spinn.py`: main file for fitting SPINN!
* `fit_<other_methods>.py`: fit using other methods

## Simulations and data analyses
Each folder specifies the simulation or data analyses. Each folder is associated with its own plotting code via `plot_<folder_name>.py`. To run the simulations/data analyses, use scons.

Note: `process_peptide_binding.py`: Processes data downloaded from https://github.com/openvax/mhcflurry

