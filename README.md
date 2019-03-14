# dp_mac

This repository contains an implementation of __DP-MAC: The Differentially Private Method of Auxiliary Coordinates for Deep Learning__, our submission to the Workshop on Privacy Preserving Machine Learning at NeurIPS2018. The paper is available: https://ppml-workshop.github.io/ppml/papers/67.pdf 

## Setup

After cloning the repository, you should be good to go! Just `cd` to the project directory and call `dp_mac_main.py`.

### Dependencies
    python 3.6
    tensorflow 1.12.0
    numpy 1.14.3
    scipy 1.1.0
    matplotlib 2.2.2 (plotting only)
    mpmath 1.0.0 (moments calculation only)


## Repository Structure

`dp_mac_train.py` runs all MAC experiments and imports functions from `dp_mac_util/`. In this directory,
- `arguments.py` contains all parameters that can be passed to `dp_mac_main.py`
- `data_handling.py` loads datasets
- `dp_losses.py` computes the losses, including clipping and noise addition
- `dp_pca.py` preprocesses data with a private PCA (only used in the MNIST experiment)
- `dp_sampler.py` computes sensitivities and draws new perturbations
- `mac_model.py` defines the model for feed-forward activation, Z-steps and (private) W-steps
- `mac_testing.py` contains functions for train and test-set validation and general logging
- `mac_training.py` contains functions for training and initialization
- `optimizers.py` provides optimizers for Z and W and adaptive learning rates for W
- `plotting.py` can be used to produce plots based on the logs of a run

`dp_sgd_autoencoder.py` implements our DP-SGD autoencoder experiment

`moments_calculation/` contains the code used for computing the privacy guarantee epsilon values, given a specific delta and sigma.
- `calculate_moments_accountant.py` is taken from repository `https://github.com/tensorflow/models/tree/master/research/differential_privacy` which was uploaded by Abadi et al. and which has since been removed
- `main_calculate_moments.py` provides an interface to call the moments calculation functions

`data/usps/usps_all.mat` contains the USPS dataset downloaded from [https://cs.nyu.edu/~roweis/data.html](https://cs.nyu.edu/~roweis/data.html) (MNIST is loaded through TensorFlow)


## Running experiments

See `dp_mac_utils/arguments.py` for all arguments taken by `dp_mac_main.py` The following settings reproduce our results:

### DP-MAC classifier
- epsilon=8
`python3.6 dp_mac_main.py -setup DP-CL -ep 30 -bs 1000 -zit 30 -pca 60 -tdt 0.3 -pca_sigma 4. -dp 1.0 -wlr 1e-2 -zlr 3e-3 -wlrdecay 0.95 -stop_decay 20 -name test_cl_eps8`

- epsilon=2
`python3.6 dp_mac_main.py -setup DP-CL -ep 30 -bs 1000 -zit 30 -pca 60 -tdt 0.3 -pca_sigma 8. -dp 2.8 -wlr 1e-2 -zlr 3e-3 -wlrdecay 0.95 -stop_decay 20 -name test_cl_eps2`

- epsilon=0.5
`python3.6 dp_mac_main.py -setup DP-CL -ep 10 -bs 500 -zit 30 -pca 60 -tdt 0.3 -pca_sigma 16. -dp 4.4 -wlr 0.03 -zlr 0.01 -wlrdecay 0.70 -stop_decay 7 -name test_cl_eps0_5`


### DP-MAC Autoencoder
- epsilon=8
`python3.6 dp_mac_main.py -setup DP-AE -ep 50 -bs 500 -zit 30 -wlr 3e-3 -wlrdecay .97 -tdt 1e-3 -zlr 1e-3 -dp 1.8 -name test_ae_eps8`

- epsilon=4
`python3.6 dp_mac_main.py -setup DP-AE -ep 50 -bs 500 -zit 30 -wlr 3e-3 -wlrdecay .97 -tdt 1e-3 -zlr 1e-3 -dp 3.1 -name test_ae_eps4`

- epsilon=2
`python3.6 dp_mac_main.py -setup DP-AE -ep 50 -bs 250 -zit 30 -wlr 3e-3 -wlrdecay .97 -tdt 1e-3 -zlr 1e-3 -dp 4.1 -name test_ae_eps2`

- epsilon=1
`python3.6 dp_mac_main.py -setup DP-AE -ep 50 -bs 250 -zit 30 -wlr 3e-3 -wlrdecay .97 -tdt 1e-3 -zlr 1e-3 -dp 7.8 -name test_ae_eps1`


### DP-SGD Autoencoder
- epsilon=8
`python3.6 dp_sgd_autoencoder.py -ep 100 -bs 500 -lr 100. -decay .99 -tg .01 -dp 2.4 -name test_dpsgd_eps8`

- epsilon=4
`python3.6 dp_sgd_autoencoder.py -ep 100 -bs 500 -lr 100. -decay .99 -tg .01 -dp 4.3 -name test_dpsgd_eps4`

- epsilon=2
`python3.6 dp_sgd_autoencoder.py -ep 100 -bs 250 -lr 50. -decay .96 -tg .01 -dp 5.7 -name test_dpsgd_eps2`

- epsilon=1
`python3.6 dp_sgd_autoencoder.py -ep 100 -bs 250 -lr 30. -decay .96 -tg .01 -dp 11. -name test_dpsgd_eps1`


If you have any questions or comments, please don't hesitate to contact [Frederik Harder](https://ei.is.tuebingen.mpg.de/employees/fharder) and [Mijung Park](https://ei.is.mpg.de/~mpark).
