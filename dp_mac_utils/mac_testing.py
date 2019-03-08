# __author__ = 'frederik harder'
import numpy as np
import tensorflow as tf
import os


class Logger:

    def __init__(self):
        self.writer = None
        self.ops = {}
        self.acc_loss_pl = None

    def init_tf(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)
        self.acc_loss_pl = tf.placeholder(dtype=tf.float32, name='loss_accumulator')

    def feed_ops(self, x_in, w_list, reconstruction, model_type):
        log_ops = self.ops
        acc_loss_pl = self.acc_loss_pl

        if model_type == 'autoencoder':
            log_ops['src_image'] = tiled_image_summary_op(x_in, 'src_image')
            log_ops['rec_image'] = tiled_image_summary_op(reconstruction, 'rec_image')

        if model_type == 'classifier':
            log_ops['train_accuracy'] = tf.summary.scalar('accuracy/train', acc_loss_pl)
            log_ops['test_accuracy'] = tf.summary.scalar('accuracy/test', acc_loss_pl)

        log_ops['test_loss'] = tf.summary.scalar('reconstruction_error/test', acc_loss_pl)
        log_ops['train_loss'] = tf.summary.scalar('reconstruction_error/train', acc_loss_pl)
        log_ops['z_hid_losses'] = [tf.summary.scalar('z_loss/hidden_{}'.format(k), acc_loss_pl)
                                   for k in range(len(w_list))]
        log_ops['z_full_loss'] = tf.summary.scalar('z_loss/full', acc_loss_pl)


logger = Logger()


def tiled_image_summary_op(t_batch, summary_name, elems=4):
    """
    plot a grid of reconstruction/target images
    :param t_batch:
    :param summary_name:
    :param elems:
    :return:
    """
    img_hw = int(np.sqrt(t_batch.shape[1].value))
    assert img_hw**2 == t_batch.shape[1].value
    tiles_hw = int(np.sqrt(elems))
    assert tiles_hw**2 == elems

    t_batch = tf.transpose(tf.reshape(t_batch[:elems, ...], shape=(elems, img_hw, img_hw, 1)), perm=(0, 2, 1, 3))
    t_imgs = tf.concat([tf.concat(tf.split(k, tiles_hw), axis=1) for k in tf.split(t_batch, tiles_hw)], axis=2)

    t_imgs = tf.clip_by_value(t_imgs, clip_value_min=0., clip_value_max=1.) * 255
    t_img = tf.cast(t_imgs, tf.uint8)
    return tf.summary.image(summary_name, t_img)


def log_z_test_batch_opt(sess, z_opt_vars_init_op, z_update_ops, z_optim, losses,
                         partial_z_loss_acc, total_z_loss_acc, args):
    """
    optimization function to log z-step convergence
    :param sess:
    :param z_opt_vars_init_op:
    :param z_update_ops:
    :param z_optim:
    :param losses:
    :param partial_z_loss_acc:
    :param total_z_loss_acc:
    :param args:
    :return:
    """
    sess.run([z_opt_vars_init_op] + z_update_ops)
    for step in range(args.num_z_steps):
        sess.run(z_optim)

        z_loss_evals = sess.run(losses.z_opt_list)
        full_loss = sess.run(losses.z_opt_total)

        for idx, z_loss in enumerate(z_loss_evals):
            partial_z_loss_acc[step][idx] += z_loss
        total_z_loss_acc[step] += full_loss

    return partial_z_loss_acc, total_z_loss_acc


def log_z_test_prep(z_list, log_dir, epoch, args):
    """
    preparation function to log z-step convergence
    :param z_list:
    :param log_dir:
    :param epoch:
    :param args:
    :return:
    """
    total_z_loss_acc = [0] * args.num_z_steps
    partial_z_loss_acc = [[0] * (len(z_list) + 1) for _ in range(args.num_z_steps)]
    epoch_writer = tf.summary.FileWriter(os.path.join(log_dir, 'epochs/{}'.format(epoch)))
    return total_z_loss_acc, partial_z_loss_acc, epoch_writer


def log_z_test_write(sess, epoch_writer, partial_z_loss_acc, total_z_loss_acc, args):
    """
    write logs on z-step convergence
    :param sess:
    :param epoch_writer:
    :param partial_z_loss_acc:
    :param total_z_loss_acc:
    :param args:
    :return:
    """
    for step in range(args.num_z_steps):
        for loss, sum_op in zip(partial_z_loss_acc[step], logger.ops['z_hid_losses']):
            sum_string = sess.run(sum_op,
                                  feed_dict={logger.acc_loss_pl: loss * args.batch_size / args.test_set_size})
            epoch_writer.add_summary(sum_string, step)
        sum_string = sess.run(logger.ops['z_full_loss'],
                              feed_dict={logger.acc_loss_pl: total_z_loss_acc[step] *
                                         args.batch_size / args.test_set_size})
        epoch_writer.add_summary(sum_string, step)


def log_test_write(sess, n_loss_acc, epoch, args):
    """
    write general test logs
    :param sess:
    :param n_loss_acc:
    :param epoch:
    :param args:
    :return:
    """
    log_ops = logger.ops
    writer = logger.writer
    test_error = n_loss_acc * args.batch_size / args.test_set_size
    test_summary_string = sess.run(log_ops['test_loss'], feed_dict={logger.acc_loss_pl: test_error})
    writer.add_summary(test_summary_string, epoch)
    print('validation error:    {:.5}'.format(test_error))

    if args.log_test_stats and 'src_image' in log_ops and 'rec_image' in log_ops:
        src_img_string, rec_img_string = sess.run([log_ops['src_image'], log_ops['rec_image']])
        writer.add_summary(src_img_string, epoch)
        writer.add_summary(rec_img_string, epoch)

    writer.flush()
    return test_error


def train_set_validation(data_loader, z_train_prep_ops, train_error_list, losses, epoch, sess, args):
    """
    run an epoch over train set and write logs
    :param data_loader:
    :param z_train_prep_ops:
    :param train_error_list:
    :param losses:
    :param epoch:
    :param sess:
    :param args:
    :return:
    """
    sess.run(data_loader.iter.initializer, feed_dict=data_loader.train_dict)
    n_loss_acc, accuracy_acc = 0, 0

    while True:
        try:
            sess.run(data_loader.get_next_op)
        except tf.errors.OutOfRangeError:
            train_acc_factor = args.batch_size / args.train_set_size
            train_error_nested = n_loss_acc * train_acc_factor
            test_summary_string = sess.run(logger.ops['train_loss'],
                                           feed_dict={logger.acc_loss_pl: train_error_nested})
            logger.writer.add_summary(test_summary_string, epoch)
            print('train error nested:  {:.5}'.format(train_error_nested))

            if losses.n_correct is not None:
                accuracy = accuracy_acc / args.train_set_size
                print('train accuracy:      {:.3}'.format(accuracy))
                train_error_list.append(accuracy)
                train_acc_string = sess.run(logger.ops['train_accuracy'],
                                            feed_dict={logger.acc_loss_pl: accuracy})
                logger.writer.add_summary(train_acc_string, epoch)

            else:
                train_error_list.append(train_error_nested)
            break

        sess.run(z_train_prep_ops)

        if losses.n_correct is None:
            nlu = sess.run(losses.nested)
        else:
            nlu, acc = sess.run([losses.nested, losses.n_correct])
            accuracy_acc += acc
        n_loss_acc = n_loss_acc + nlu


def test_set_validation(data_loader, z_list, z_opt_vars_init_op, z_optim, z_update_ops, log_dir,
                        test_error_list, losses, epoch, sess, args):
    """
    run an epoch over test set and write logs
    :param data_loader:
    :param z_list:
    :param z_opt_vars_init_op:
    :param z_optim:
    :param z_update_ops:
    :param log_dir:
    :param test_error_list:
    :param losses:
    :param epoch:
    :param sess:
    :param args:
    :return:
    """
    sess.run(data_loader.iter.initializer, feed_dict=data_loader.test_dict)
    n_loss_acc, accuracy_acc = 0, 0
    if args.log_z_opt:
        total_z_loss_acc, partial_z_loss_acc, epoch_writer = log_z_test_prep(z_list, log_dir, epoch, args)
    else:
        total_z_loss_acc, partial_z_loss_acc, epoch_writer = None, None, None
    while True:
        try:
            sess.run(data_loader.get_next_op)
        except tf.errors.OutOfRangeError:
            test_error = log_test_write(sess, n_loss_acc, epoch, args)
            if args.log_z_opt:
                log_z_test_write(sess, epoch_writer, partial_z_loss_acc, total_z_loss_acc, args)

            if losses.n_correct is not None:
                accuracy = accuracy_acc / args.test_set_size
                print('validation accuracy: {}'.format(accuracy))
                train_acc_string = sess.run(logger.ops['test_accuracy'],
                                            feed_dict={logger.acc_loss_pl: accuracy})
                logger.writer.add_summary(train_acc_string, epoch)
                test_error_list.append(accuracy)
            else:
                test_error_list.append(test_error)
            break

        if losses.n_correct is None:
            nlu = sess.run(losses.nested)
        else:
            nlu, acc = sess.run([losses.nested, losses.n_correct])
            accuracy_acc += acc
        n_loss_acc += nlu
        if args.log_z_opt:
            z_log_args = [sess, z_opt_vars_init_op, z_update_ops, z_optim, losses,
                          partial_z_loss_acc, total_z_loss_acc, args]
            partial_z_loss_acc, total_z_loss_acc = log_z_test_batch_opt(*z_log_args)

    print('-------------------------------------------')


def log_args(log_dir, args):
    """ print and save all args """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args_log'), 'w') as f:
        lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
        f.writelines(lines)
        for line in lines:
            print(line.rstrip())
    print('-------------------------------------------')


def log_errors(train_errors, test_errors, log_dir):
    """
    store arrays with nested error vals per epoch
    """
    assert os.path.exists(log_dir)
    np.save(os.path.join(log_dir, 'train_errors.npy'), np.asarray(train_errors))
    np.save(os.path.join(log_dir, 'test_errors.npy'), np.asarray(test_errors))
    print(np.asarray(train_errors))
    print(np.asarray(test_errors))
