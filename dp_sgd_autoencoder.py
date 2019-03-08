# __author__ = 'frederik harder'
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import argparse
import os


basedir = '/home/'  # enter project directory


def update_dp_sgd_op(loss, z_list, w_list, h_list, sigma, lr, layerwise_clip, max_bound=4.):

    # get gradients wrt. z_list and w_list
    z_grads = tf.gradients(loss, z_list)  # (bs, d_layer) x n_layers

    # (bs, d_layer, 1) x (bs, 1, d_out) -> (bs, d_layer, d_out)
    w_ps_grads = [tf.matmul(h[:, :, None], z[:, None, :]) for h, z in zip(h_list, z_grads)]

    # clip w_grads by norm
    if max_bound is not None:
        w_ps_grads = clip_grads(w_ps_grads, max_bound, layerwise_clip)

    # sum over samples
    w_grads = [tf.reduce_mean(w, axis=0) for w in w_ps_grads]

    # add noise
    if max_bound and sigma:
        sdevs = get_sdevs(sigma, max_bound, w_ps_grads, layerwise_clip)
        w_grads = [g + tf.random_normal(g.get_shape(), stddev=s) for g, s in zip(w_grads, sdevs)]

    # apply w_grads, return op
    w_update_ops = [tf.assign_sub(w, lr * g) for w, g in zip(w_list, w_grads)]

    return tf.group(w_update_ops)


def clip_grads(w_ps_grads, max_bound, layerwise_clip):
    quad_sums = [tf.reduce_sum(w ** 2, axis=[1, 2]) for w in w_ps_grads]  # (bs) x n_layers

    # avg_grad_norms = [tf.reduce_mean(tf.sqrt(s)) for s in quad_sums]
    # agn_print_op = tf.print(avg_grad_norms)

    if layerwise_clip:
        grad_norms = [tf.sqrt(s) for s in quad_sums]
        norm_factors = [tf.minimum(max_bound / n, tf.ones(n.get_shape())) for n in grad_norms]  # (bs)
        w_ps_grads = [w * f[:, None, None] for w, f in zip(w_ps_grads, norm_factors)]
    else:
        grad_norms = tf.sqrt(tf.add_n(quad_sums))
        norm_factors = tf.minimum(max_bound / grad_norms, tf.ones(grad_norms.get_shape()))  # (bs)
        # with tf.control_dependencies([agn_print_op]):
        w_ps_grads = [w * norm_factors[:, None, None] for w in w_ps_grads]
    return w_ps_grads


def get_sdevs(sigma, max_bound, w_ps_grads, layerwise_clip):
    bs = w_ps_grads[0].get_shape()[0].value
    n_layers = len(w_ps_grads)
    if layerwise_clip:
        if isinstance(max_bound, list):
            sdevs = [sigma * b * np.sqrt(n_layers) / (2 * bs) for b in max_bound]
        else:
            sdev = sigma * max_bound * np.sqrt(n_layers) / (2 * bs)
            sdevs = [sdev] * n_layers
    else:
        sdev = sigma * max_bound / (2 * bs)
        sdevs = [sdev] * n_layers
    return sdevs


def usps_data(x_bound=None):
    file_path = os.path.join(basedir, 'data/usps/usps_all.mat')
    train_set_size = 5000
    test_set_size = 5000

    source_mat = loadmat(file_path)['data']
    train_per_label = train_set_size // 10
    test_per_label = test_set_size // 10
    train_data, test_data = [], []
    for label in range(10):
        label_data = source_mat[:, :, label].T
        # the dataset contains 11k (1.1k per class) images in total. we select 5k (500 p.c.) for train an test randomly
        subset = np.random.choice(np.arange(1100), train_per_label + test_per_label, replace=False)
        data_select = label_data[subset, :]
        train_data.append(data_select[:train_per_label, :])
        test_data.append(data_select[train_per_label:, :])
    train_data = np.concatenate(train_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    # shuffle
    perm = np.random.permutation(train_set_size)
    train_data = train_data[perm, :]

    perm = np.random.permutation(test_set_size)
    test_data = test_data[perm, :]

    # set to float
    train_data = train_data.astype(np.float32) / 255.
    test_data = test_data.astype(np.float32) / 255.

    train_data = train_data - train_data.mean()
    test_data = test_data - test_data.mean()

    if x_bound is not None:
        train_norms = np.linalg.norm(train_data, 2, axis=1)
        test_norms = np.linalg.norm(test_data, 2, axis=1)
        max_norm = np.maximum(np.max(train_norms), np.max(test_norms))  # 12.496 for USPS
        x_scaling = np.minimum(x_bound / max_norm, 1.)
        train_data = train_data * x_scaling
        test_data = test_data * x_scaling
    else:
        x_scaling = 1.

    return train_data, test_data, x_scaling


def get_usps_iter(bs):
    x_trn, x_tst, x_scaling = usps_data()

    data_x_pl = tf.placeholder(tf.float32, shape=[None, 256])
    dataset = tf.data.Dataset.from_tensor_slices(tuple([data_x_pl]))
    dataset = dataset.batch(bs)
    data_iter = dataset.make_initializable_iterator()
    x_iter = data_iter.get_next()[0]
    set_load_op = data_iter.initializer
    train_dict = {data_x_pl: x_trn}
    test_dict = {data_x_pl: x_tst}
    return x_iter, set_load_op, train_dict, test_dict, x_scaling


def model(img, bs):
    dims = [256, 300, 100, 20, 100, 300, 256]
    h0 = tf.concat([img, tf.ones((bs, 1))], axis=1)

    w_list = []
    z_list = []
    h_list = [h0]

    h = h0

    for idx in range(1, len(dims)):
        d_in = dims[idx-1]
        d_out = dims[idx]
        with tf.variable_scope('fc{}'.format(idx)):
            w = tf.get_variable('w', (d_in+1, d_out), initializer=tf.glorot_normal_initializer(), dtype=tf.float32)
            z = tf.matmul(h, w)

            # pre_h = tf.nn.softplus(z)
            pre_h = tf.nn.relu(z) if idx < len(dims)-1 else z
            h = tf.concat([pre_h, tf.ones((bs, 1))], axis=1) if idx < len(dims)-1 else pre_h

            w_list.append(w)
            z_list.append(z)
            h_list.append(h)

    return w_list, z_list, h_list


def get_loss(pred, tgt):
    mse_loss = 0.5 * tf.reduce_mean(tf.reduce_sum((tgt - pred) ** 2, axis=-1))
    return mse_loss


def train_dpsgd_ae(bs, n_epochs, lr, lr_decay, sigma, layerwise_clip, max_bound):
    # get mnist
    x_iter, set_load_op, train_dict, test_dict, _ = get_usps_iter(bs)

    w_list, z_list, h_list = model(x_iter, bs)
    loss = get_loss(h_list[-1], x_iter)

    lr_pl = tf.placeholder(dtype=tf.float32, shape=())

    w_update_op = update_dp_sgd_op(loss, z_list, w_list, h_list, sigma, lr_pl, layerwise_clip, max_bound)

    train_log = []
    test_log = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(n_epochs):
            sess.run(set_load_op, feed_dict=train_dict)

            while True:
                try:
                    sess.run(w_update_op, feed_dict={lr_pl: lr * lr_decay ** ep})
                except tf.errors.OutOfRangeError:
                    break

            sess.run(set_load_op, feed_dict=train_dict)

            acc_mse = 0
            count = 0
            while True:
                try:
                    acc_mse += sess.run(loss)
                    count += 1
                except tf.errors.OutOfRangeError:
                    if ep % 10 == 0 or ep == n_epochs - 1:
                        print('{} trn mse: {}'.format(ep, acc_mse / count))
                    train_log.append(acc_mse / count)
                    break

            sess.run(set_load_op, feed_dict=test_dict)

            acc_mse = 0
            count = 0
            while True:
                try:
                    acc_mse += sess.run(loss)
                    count += 1
                except tf.errors.OutOfRangeError:
                    if ep % 5 == 0 or ep == n_epochs - 1:
                        print('{} tst mse: {}'.format(ep, acc_mse / count))
                    test_log.append(acc_mse / count)
                    break
    return train_log, test_log


def save_logs(train_log, test_log, name):
    save_dir = os.path.join(basedir, 'results/{}'.format(name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'train_log.npy'), train_log)
    np.save(os.path.join(save_dir, 'test_log.npy'), test_log)
    print('saved logs for run: {}'.format(name))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', '-name', type=str, default='dp_sgd_test')  # will save logs under this name
    parser.add_argument('--num_epochs', '-ep', type=int, default=None)  # number of training epochs
    parser.add_argument('--batch_size', '-bs', type=int, default=None)  # batch size
    parser.add_argument('--learning_rate', '-lr', type=float, default=None)
    parser.add_argument('--lr_decay', '-decay', type=float, default=None)
    parser.add_argument('--sigma', '-dp', type=float, default=None)
    parser.add_argument('--max_bound', '-tg', type=float, default=None)
    parser.add_argument('--layerwise_clip', dest='layerwise_clip', action='store_true')
    return parser.parse_args()


def main():
    args = parse_arguments()
    train_log, test_log = train_dpsgd_ae(args.batch_size, args.num_epochs, args.learning_rate,
                                         args.lr_decay, args.sigma, args.layerwise_clip, args.max_bound)
    save_logs(train_log, test_log, args.experiment_name)


if __name__ == '__main__':
    main()
