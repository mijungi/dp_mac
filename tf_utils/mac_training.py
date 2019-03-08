import numpy as np
import tensorflow as tf
from tf_utils.mac_testing import logger


def opt_var_init_ops(global_w_step, w_list):
    """
    creates initialization ops for all variables that need it
    :param global_w_step: count for learning rate decay
    :param w_list list of weights
    :return: z_optimizer initializer (which resets on every batch) and a general initializer called once
    """
    # retrieve parameters stored by optimizers
    z_opt_vars = [k for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if k.name.startswith('z_opt')]
    w_opt_vars = [k for k in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if k.name.startswith('w_opt')]

    z_opt_vars_init_op = tf.variables_initializer(z_opt_vars)

    vars_to_init = w_opt_vars + w_list

    if global_w_step is not None:
            vars_to_init.append(global_w_step)

    extra_vars_init_op = tf.variables_initializer(vars_to_init)
    return z_opt_vars_init_op, extra_vars_init_op


def train_batch(optims, draw_pert_op, train_batch_count, sess, args):
    """
    perform z-steps, w-steps, noise-draws and clipping according to the chosen optimizers
    :param sess: session
    :param optims: optimizer tuple
    :param draw_pert_op: drawing new noise
    :param train_batch_count:
    :param args: run arguments
    :return:
    """
    for _ in range(args.num_z_steps):  # Z-step
        sess.run(optims.z)

    sess.run(draw_pert_op)  # draw fresh noise

    if args.log_w_opt:
        w_step_logging(sess, train_batch_count)
    sess.run(optims.w)  # W-step


def train_epoch(it_count, data_loader, z_opt_vars_init_op, z_update_ops, optims,
                sigmas, draw_pert_op, wlr, epoch, sess, args):
    """
    performs one epoch of training along with printing errors,
    learning rates and sigmas, storing and loading zs, etc.
    all arguments as defined in tf_dp_mac main()
    """

    sess.run(data_loader.iter.initializer, feed_dict=data_loader.train_dict)
    print('epoch {}:'.format(epoch))
    if isinstance(wlr, tf.Tensor):
        print('W learning rate: {:.5}'.format(sess.run(wlr)))
    if sigmas[0] is not None and (args.dt_bound is None or epoch == 0):
        ep_sigmas = sess.run(sigmas) if not isinstance(sigmas[0], np.float64) else sigmas
        for idx, sig in enumerate(ep_sigmas):
            print('layer {} sigma:   {:.5}'.format(idx, sig))
    print('')

    approx_it_per_epoch = args.train_set_size / args.batch_size
    tgt_it = it_count + approx_it_per_epoch
    if tgt_it > args.num_iterations:
        tgt_it = args.num_iterations

    while it_count < tgt_it:
        sess.run(data_loader.get_next_op)
        it_count += 1
        sess.run([z_opt_vars_init_op] + z_update_ops)
        train_batch(optims, draw_pert_op, it_count, sess, args)

    return it_count


def w_step_logging(sess, train_batch_count):
    batch_log_ops = ('hist_dt_norms', 'median_dt_norms')
    for op_name in batch_log_ops:
        if op_name in logger.ops:
            log_ops = logger.ops[op_name]
            if not isinstance(log_ops, list):
                log_ops = [log_ops]
            if isinstance(log_ops[0], list):
                log_ops = [k[i] for k in logger.ops[op_name] for i in range(len(k)) if k[i] is not None]
            else:
                log_ops = [k for k in log_ops if k is not None]
            log_strings = sess.run(log_ops)
            for log in log_strings:
                logger.writer.add_summary(log, train_batch_count)
