# __author__ = 'frederik harder'
import argparse
import numpy as np


def parse_arguments(direct_feed=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', '-name', type=str, default='tf_test')  # will save logs under this name
    parser.add_argument('--random_seed', '-seed', type=int, default=None)  # set seed. will draw seed if None
    parser.add_argument('--named_setup', '-setup', type=str, default=None)  # sets all other args to special setup

    # model specs
    parser.add_argument('--model_type', '-model', type=str, default='autoencoder')  # also supports 'classifier'
    parser.add_argument('--topology', '-top', type=str, default='300,100,20,100,300')  # list of hidden layer sizes

    # data
    parser.add_argument('--dataset', '-data', type=str, default='usps')  # also suppports 'mnist'
    parser.add_argument('--train_set_size', '-trset', type=int, default=None)  # defaults to appropriate max size ...
    parser.add_argument('--test_set_size', '-teset', type=int, default=None)  # subsamples otherwise

    # training
    parser.add_argument('--num_epochs', '-ep', type=int, default=None)  # number of training epochs (approximate)
    parser.add_argument('--batch_size', '-bs', type=int, default=None)  # batch size

    # optimizers
    parser.add_argument('--w_optim', '-wopt', type=str, default=None)  # options: adam, sgd, mom(entum)
    parser.add_argument('--z_optim', '-zopt', type=str, default=None)  # options: same
    parser.add_argument('--w_learning_rate', '-wlr', type=float, default=None)
    parser.add_argument('--z_learning_rate', '-zlr', type=float, default=None)
    parser.add_argument('--w_lr_decay', '-wlrdecay', type=float, default=None)  # lr-percentage retained per epoch ...
    parser.add_argument('--lr_decay_stop', '-stop_decay', type=int, default=None)  # stops decreasing lr after n epochs
    parser.add_argument('--num_z_steps', '-zit', type=int, default=None)  # number of z steps

    # constraints, noise level
    parser.add_argument('--dt_bound', '-tdt', type=str, default=None)  # clip delta T in linear loss component b
    parser.add_argument('--dp_sigma', '-dp', type=float, default=None)  # base standard deviation for gaussian noise

    # DP-PCA
    parser.add_argument('--dp_pca_dims', '-pca', type=int, default=None)
    parser.add_argument('--dp_pca_sigma', '-pca_sigma', type=float, default=None)

    # logging
    parser.add_argument('--result_dir', type=str, default='../results')  # directory for storing results
    parser.add_argument('--log_w_opt', dest='log_w_opt', action='store_true')  # if true, make logs during w-step
    parser.add_argument('--log_z_opt', dest='log_z_opt', action='store_true')  # if true, logs z step convergence
    parser.add_argument('--log_train_stats', dest='log_train_stats', action='store_true')  # if true, logs train-eval
    parser.add_argument('--log_test_stats', dest='log_test_stats', action='store_true')  # if true, logs test-eval
    parser.add_argument('--log_b_noise_ratio', dest='log_b_noise_ratio', action='store_true')  # log b and eps per batch
    parser.add_argument('--log_all', dest='log_all', action='store_true')  # if true, logs everything
    parser.add_argument('--save_weights', dest='save_weights', action='store_true')

    args = parser.parse_args(direct_feed)

    args = handle_special_setups(args)
    args = enforce_constraints(args)

    return args


def enforce_constraints(args):
    """
    - ensures that a number of constraints on the arguments are met and keywords are correct
    - re-assigns some complex args by keyword (topology, regularizers, z_bound)
    - chooses some defaults depending on other args (act_out, dataset sizes, seed, x_bound)
    """
    # important keyword ranges
    assert args.model_type in ('autoencoder', 'classifier')
    assert args.dataset in ('usps', 'mnist', 'pretrained_cifar') or args.dataset.startswith('cifar_feats')

    # constraints
    assert not (args.model_type == 'classifier') or args.topology != '300,100,20,100,300'  # set classifier topology
    assert not (args.dataset == 'usps' and args.model_type == 'classifier')  # we don't use usps for classification

    # assign complex values based on keywords
    assert isinstance(args.topology, str)
    args.topology = [int(k) for k in args.topology.split(',')]

    assert isinstance(args.dt_bound, str)
    n_layers = len(args.topology) + 1
    args.dt_bound = get_layer_list(args.dt_bound, n_layers)

    # assign defaults
    if args.train_set_size is None:
        trn_sizes = {'usps': 5000, 'mnist': 60000, 'pretrained_cifar': 50000}
        args.train_set_size = trn_sizes[args.dataset]
    if args.test_set_size is None:
        tst_sizes = {'usps': 1000, 'mnist': 10000, 'pretrained_cifar': 10000}
        args.test_set_size = tst_sizes[args.dataset]

    if args.random_seed is None:
        args.random_seed = np.random.randint(0, 10000)

    if args.log_all:
        args.log_z_opt = True
        args.log_w_opt = True
        args.log_train_stats = True
        args.log_test_stats = True

    args.num_iterations = args.train_set_size / args.batch_size * args.num_epochs

    return args


def get_layer_list(param, n_layers):
    if param is None:
        param = [None] * n_layers
    elif isinstance(param, str) and ',' in param:
        param = [float(k) for k in param.split(',')]
    elif isinstance(param, str):
        param = [float(param)] * n_layers
    else:
        raise ValueError
    return param


def handle_special_setups(args):
    name = args.named_setup
    if name is None:
        return args

    setups = dict()

    # args to set for DP-AE: -name, -name, -ep, -bs, -wlr, -zlr, -wlrdecay, -zit, -tdt -dp
    setups['DP-AE'] = [('model_type', 'autoencoder'), ('topology', '300,100,20,100,300'), ('dataset', 'usps'),
                       ('w_optim', 'adam'), ('z_optim', 'adam')]

    # args to set for DP-MLP: -name, -ep, -bs, -wlr, -zlr, -wlrdecay, -zit, -tdt -dp
    setups['DP-MLP'] = [('model_type', 'classifier'), ('topology', '300'), ('dataset', 'mnist'),
                        ('w_optim', 'adam'), ('z_optim', 'adam')]

    assert name in setups.keys()

    for attr, val in setups[name]:
        setattr(args, attr, val)

    return args
