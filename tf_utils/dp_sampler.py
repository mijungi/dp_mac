import numpy as np
import tensorflow as tf


def get_eps_mats(w_list):

    eps_list = []
    draw_ops = []
    for idx, w in enumerate(w_list):
        eps_shape = w.get_shape()
        eps = tf.get_variable('eps_l{}'.format(idx), shape=eps_shape)
        draw_ops.append(tf.assign(eps, tf.random_normal(eps_shape)))
        eps_list.append(eps)
    return eps_list, tf.group(draw_ops)


def get_layer_perturbation(w, tdt, layer_idx, args, partition_penalty=True):
    """
    unifies get_eps_mats and get_perturbations per layer
    :param w: layer weight
    :param tdt
    :param layer_idx
    :param args:
    :param partition_penalty
    :return: perturbation, pert_sigma, draw_eps_op, sensitivity
    """
    if args.dp_sigma is None:
        return None, None, tf.no_op(), None

    eps_shape = w.get_shape()
    n_layers = len(args.topology) + 1

    assert tdt is not None
    sens = compute_sensitivities_dt(tdt, args.batch_size)

    if partition_penalty:
        sens *= np.sqrt(n_layers)  # partitioning released vector into K parts causes sqrt(K) noise increase

    pert_sigma = args.dp_sigma * sens

    # drawing epsilon
    pert = tf.get_variable('pert_l{}'.format(layer_idx), shape=eps_shape)
    draw_eps_op = tf.assign(pert, tf.random_normal(eps_shape, stddev=pert_sigma))

    return pert, pert_sigma, draw_eps_op, sens


def compute_sensitivities_dt(td, batch_size):
    """
    bounds gradient sensitivity if each delta T_kn in b_k is bounded by td (independent of original loss function)
    """
    return td / (2 * batch_size)
