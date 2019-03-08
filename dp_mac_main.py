# __author__ = 'frederik harder'
import numpy as np
import tensorflow as tf
import os
from collections import namedtuple

from dp_mac_utils.arguments import parse_arguments
from dp_mac_utils.data_handling import get_data_loader
from dp_mac_utils.mac_model import get_params, fwd_model, z_mac_model, init_zs_fwd, w_mac_model
from dp_mac_utils.mac_testing import train_set_validation, test_set_validation, log_args, log_errors, logger
from dp_mac_utils.optimizers import get_optimizers
from dp_mac_utils.mac_training import opt_var_init_ops, train_epoch


def main():
    """
    builds model according to arguments and trains if for a number of epochs while logging various statistics
    """
    args = parse_arguments()

    log_dir = os.path.join(args.result_dir, args.experiment_name)
    log_args(log_dir, args)  # log and print arguments
    logger.init_tf(log_dir)  # get logger ready to accept ops

    # random seed
    if args.random_seed is not None:
        tf.set_random_seed(args.random_seed)
        np.random.seed(args.random_seed)

    # data
    xy_var, data_loader = get_data_loader(args)
    args.n_feats = xy_var.x.get_shape()[1].value
    args.n_labels = None if args.model_type == 'autoencoder' else xy_var.y.get_shape()[1].value

    z_list, w_list = get_params(args)  # parameters

    # model and losses
    reconstruction, hidden_preds, nested_loss, n_correct = fwd_model(xy_var, w_list)
    z_opt_loss, z_losses = z_mac_model(xy_var, w_list, z_list)
    w_opt_losses, draw_pert_op, sigmas = w_mac_model(xy_var, w_list, z_list, args)

    loss_collection = namedtuple('losses', ['nested', 'n_correct', 'w_opt_list', 'z_opt_list', 'z_opt_total'])
    losses = loss_collection(nested_loss, n_correct, w_opt_losses, z_losses, z_opt_loss)

    z_update_ops = init_zs_fwd(hidden_preds, z_list)  # z updates

    logger.feed_ops(xy_var.x, w_list, reconstruction, args.model_type)  # most logging ops

    optims, global_w_step, wlr = get_optimizers(w_list, z_list, losses, args)  # optimizers

    z_opt_vars_init_op, extra_vars_init_op = opt_var_init_ops(global_w_step, w_list)  # variable init functions

    # prep for saving errors per epoch and weights after training
    train_error_list, test_error_list = [], []
    saver = tf.train.Saver(w_list) if args.save_weights else None

    with tf.Session() as sess:  # start session
        sess.run(extra_vars_init_op)  # initialization

        # start loop
        it_count, epoch = 0, 0
        while it_count < args.num_iterations:
            it_count = train_epoch(it_count, data_loader, z_opt_vars_init_op, z_update_ops, optims,
                                   sigmas, draw_pert_op, wlr, epoch, sess, args)

            train_set_validation(data_loader, z_update_ops, train_error_list, losses, epoch, sess, args)

            test_set_validation(data_loader, z_list, z_opt_vars_init_op, optims.z, z_update_ops,
                                log_dir, test_error_list, losses, epoch, sess, args)
            epoch += 1

        # save weights
        if saver is not None:
            saver.save(sess, '{}/{}/weights'.format(args.result_dir, args.experiment_name), write_meta_graph=False)

    log_errors(train_error_list, test_error_list, log_dir)  # save errors


if __name__ == '__main__':
    main()
