# __author__ = 'frederik harder'
import numpy as np
import tensorflow as tf
from dp_mac_utils.dp_losses import get_fc_mse_loss, get_fc_cce_loss
from dp_mac_utils.dp_sampler import get_layer_perturbation


def get_params(args):
    """
    creates lists of weights, zs and regularizer variables
    :param args: run arguments
    :return:
    """
    layers = args.topology

    # make z_list
    z_list = []
    for idx, l_size in enumerate(layers):
        with tf.variable_scope('layer_{}'.format(idx), reuse=tf.AUTO_REUSE):
            z_list.append(tf.get_variable('z', shape=(args.batch_size, l_size)))

    # make w_list
    def layer_w(f_in, f_out, scope_name):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            w_init = tf.random_normal(shape=(f_in + 1, f_out), stddev=np.sqrt(2 / (f_in + f_out)))
            t_w = tf.get_variable('weight_bias', initializer=w_init)
        return t_w
    w_list = []
    d_out = args.n_feats if args.n_labels is None else args.n_labels
    for idx, feat_in, feat_out in zip(range(len(layers) + 1), [args.n_feats] + layers, layers + [d_out]):
        w_list.append(layer_w(feat_in, feat_out, 'layer_{}'.format(idx)))

    return z_list, w_list


def fwd_model(xy_var, w_list):
    """
    computes simple forward pass of the model
    :param xy_var: in and out variables
    :param w_list: weights per layer
    :return:
    """
    x_in = xy_var.x
    preds_hid = []
    x_hid = x_in
    for w in w_list[:-1]:
        x_hid = tf.nn.relu(x_hid @ w[:-1, :] + w[-1, :])
        preds_hid.append(x_hid)

    w = w_list[-1]
    pred_out = x_hid @ w[:-1, :] + w[-1, :]

    if xy_var.y is not None:
        ce_term = tf.nn.softmax_cross_entropy_with_logits_v2(labels=xy_var.y, logits=pred_out)
        nested_loss = tf.reduce_mean(ce_term)

        accuracy_count = tf.cast(tf.equal(tf.argmax(xy_var.y, axis=1), tf.argmax(pred_out, axis=1)), tf.float32)
        n_correct = tf.reduce_sum(accuracy_count)
    else:
        nested_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((x_in - pred_out) ** 2, axis=-1))
        n_correct = None

    return pred_out, preds_hid, nested_loss, n_correct


def z_mac_model(xy_var, w_list, z_list):
    """
    computes layerwise and joint losses for z-optimization
    :param xy_var: in and out variables
    :param w_list: weights per layer
    :param z_list: zs per layer
    :return:
    """
    x_in = xy_var.x
    losses = []
    for layer, z_in, z_tgt, w in zip(range(len(z_list)), [x_in] + z_list[:-1], z_list, w_list[:-1]):
        with tf.variable_scope('layer_{}'.format(layer), reuse=tf.AUTO_REUSE):
            z_rec = tf.nn.relu(z_in @ w[:-1, :] + w[-1, :])

            loss = tf.reduce_mean(tf.reduce_sum((z_tgt - z_rec) ** 2, axis=-1))
            losses.append(loss)

    with tf.variable_scope('layer_{}'.format(len(z_list)), reuse=tf.AUTO_REUSE):
        w, z_in = w_list[-1], z_list[-1]
        z_rec = z_in @ w[:-1, :] + w[-1, :]

        if xy_var.y is not None:
            ce_term = -tf.reduce_sum(xy_var.y * tf.log_sigmoid(z_rec) + (1-xy_var.y) * tf.log_sigmoid(-z_rec), axis=-1)
            out_loss = tf.reduce_mean(ce_term)
        else:
            out_loss = tf.reduce_mean(tf.reduce_sum((x_in - z_rec) ** 2, axis=-1))

        losses.append(out_loss)
    opt_loss = tf.reduce_sum(losses)
    return opt_loss, losses


def w_mac_model(xy_var, w_list, z_list, args):
    """
    computes layerwise losses for w-optimization, including dp-setting
    :param xy_var: in and out variables
    :param w_list: weights per layer
    :param z_list: zs per layer
    :param args: run arguments
    :return: layerwise losses, ops for approximated loss and dp setting
    """
    tgt_list = z_list + [xy_var.y] if xy_var.y is not None else z_list + [xy_var.x]
    ipt_list = [xy_var.x] + z_list

    losses, draw_ops, sigmas, sensitivities = [], [], [], []
    layer_enum = list(zip(range(len(z_list) + 1), ipt_list, tgt_list, w_list, args.dt_bound))

    for idx, z_in, z_tgt, w, tdt in layer_enum:
        pert, sigma, draw_op, sens = get_layer_perturbation(w, tdt, idx, args)
        draw_ops.append(draw_op)
        sigmas.append(sigma)
        sensitivities.append(sens)

        with tf.variable_scope('layer_{}'.format(idx), reuse=tf.AUTO_REUSE):
            if idx < len(z_list):
                layer_loss = get_fc_mse_loss(z_tgt, z_in, w, pert, tdt, layer='hid')
            else:  # last layer
                if xy_var.y is not None:
                    layer_loss = get_fc_cce_loss(z_tgt, z_in, w, pert, tdt)
                else:
                    layer_loss = get_fc_mse_loss(z_tgt, z_in, w, pert, tdt, layer='out')
            losses.append(layer_loss)
    draw_pert_op = tf.group(draw_ops)

    return losses, draw_pert_op, sigmas


def init_zs_fwd(preds_hid, z_list):
    """
    :return ops to initialize zs via forward pass
    """
    init_ops = []
    for act, z in zip(preds_hid, z_list):
        init_ops.append(tf.assign(z, act))
    return init_ops
