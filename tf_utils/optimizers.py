import tensorflow as tf
from collections import namedtuple


def get_sgd_fun(kind, name, lr):
    """
    :return SGD optimizer variant indicated by kind, taking given name and learning rate
    """
    funs = {'adam': tf.train.AdamOptimizer(name=name, learning_rate=lr),
            'rmsprop': tf.train.RMSPropOptimizer(name=name, learning_rate=lr),
            'sgd': tf.train.GradientDescentOptimizer(name=name, learning_rate=lr),
            'mom': tf.train.MomentumOptimizer(name=name, learning_rate=lr, momentum=0.9)}
    assert kind in funs
    return funs[kind]


def get_learning_rates(args):
    """
    :return: constant or decaying learning rates per epoch, depending on settings
    """
    w_lr = args.w_learning_rate
    z_lr = args.z_learning_rate  # adaptive learning rate for z have not been useful
    global_w_step = None

    if args.w_lr_decay is not None:
        global_w_step = tf.get_variable('global_w_step', dtype=tf.int32, initializer=0)
        epoch_steps = args.train_set_size / args.batch_size
        if args.w_optim != 'nested':
            epoch_steps *= len(args.topology) + 1  # (n_layers)
        w_lr = tf.train.exponential_decay(w_lr, global_w_step, epoch_steps, args.w_lr_decay, staircase=True)

        if args.lr_decay_stop is not None:
            min_lr = args.w_learning_rate * args.w_lr_decay**(args.lr_decay_stop-1)  # no change after lr_decay_stop ep
            w_lr = tf.maximum(w_lr, min_lr)

    return w_lr, z_lr, global_w_step


def get_optimizers(w_list, z_list, losses, args):
    """
    create a tuple of optimizer operations depending on settings.
    also return global steps to track learning rate decay and w-learning rate for logging
    :param w_list: weights per layer
    :param z_list: zs per layer
    :param losses: tuple of all losses
    :param args: run arguments
    :return:
    """
    w_lr, z_lr, global_w_step = get_learning_rates(args)

    w_optim = get_w_optim_sgd(losses, w_list, w_lr, global_w_step, args)
    z_optim = get_z_optim_sgd(losses, z_lr, z_list, args)

    optim_collect = namedtuple('optim', ['w', 'z'])
    optims = optim_collect(w_optim, z_optim)
    return optims, global_w_step, w_lr


def get_w_optim_sgd(losses, w_list, w_lr, global_w_step, args):
    with tf.variable_scope('w_opt'):
        w_optim = []
        for idx, w, loss in zip(range(len(w_list)), w_list, losses.w_opt_list):
            w_fun = get_sgd_fun(kind=args.w_optim, name='w_optim_{}'.format(idx), lr=w_lr)
            min_op = w_fun.minimize(loss, var_list=[w], global_step=global_w_step)
            w_optim.append(min_op)
    return w_optim


def get_z_optim_sgd(losses, z_lr, z_list, args):
    with tf.variable_scope('z_opt'):
        z_fun = get_sgd_fun(kind=args.z_optim, name='z_optim', lr=z_lr)
        z_optim = z_fun.minimize(losses.z_opt_total, var_list=z_list)
    return z_optim
