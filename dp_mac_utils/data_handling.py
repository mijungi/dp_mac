# __author__ = 'frederik harder'
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from collections import namedtuple
from dp_mac_utils.dp_pca import pca_preprocess


usps_data_path = 'data/usps/usps_all.mat'

xy_data = namedtuple('xy_data', ['x', 'y'])
data_loader = namedtuple('data_loader', ['feed_sampler_op', 'iter', 'get_next_op', 'train_dict', 'test_dict'])


def get_datasets(args):
    """
    chooses right dataset to load
    """
    assert args.dataset in ('usps', 'mnist')
    return read_usps_data(args) if args.dataset == 'usps' else read_mnist_data(args)


def read_usps_data(args, file_path=usps_data_path):
    """
    loads usps dataset from matlab file
    :param args: run arguments
    :param file_path: path to usps data .mat file
    :return: train and test data tuples and the effective scale a if data has been scaled to [0,a]
    """
    assert args.train_set_size + args.test_set_size <= 11000
    source_mat = loadmat(file_path)['data']
    train_per_label = args.train_set_size // 10
    test_per_label = args.test_set_size // 10
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
    perm = np.random.permutation(args.train_set_size)
    train_data = train_data[perm, :]

    perm = np.random.permutation(args.test_set_size)
    test_data = test_data[perm, :]

    # to float
    train_data = train_data.astype(np.float32) / 255.
    test_data = test_data.astype(np.float32) / 255.

    train_data = train_data - train_data.mean()
    test_data = test_data - test_data.mean()

    train_xy_data = xy_data(train_data, None)
    test_xy_data = xy_data(test_data, None)

    return train_xy_data, test_xy_data


def read_mnist_data(args):
    """
    loads mnist data from keras library. centers it and rescales to range [-1,1] (or depending on args.x_bound)
    :param args: run arguments
    :return: train and test data tuples and the effective scale a if data has been scaled to [-a,a]
    """

    def to_one_hot(vec):
        mat = np.zeros((vec.size, vec.max() + 1))
        mat[np.arange(vec.size), vec] = 1
        return mat

    mnist = tf.keras.datasets.mnist.load_data()
    n_feats = 784
    images = np.reshape(mnist[0][0], (60000, n_feats))
    labels = to_one_hot(mnist[0][1])
    train_images = images[:args.train_set_size, :]
    train_labels = labels[:args.train_set_size]

    images = np.reshape(mnist[1][0], (10000, n_feats))
    labels = to_one_hot(mnist[1][1])
    test_images = images[:args.test_set_size, :]
    test_labels = labels[:args.test_set_size]

    if args.dp_pca_dims is not None:
        train_images, test_images = pca_preprocess(train_images, test_images, args)

    # centering
    train_images = train_images - train_images.mean()
    test_images = test_images - test_images.mean()

    get_y = (args.model_type == 'classifier')
    train_xy_data = xy_data(train_images, train_labels if get_y else None)
    test_xy_data = xy_data(test_images, test_labels if get_y else None)

    return train_xy_data, test_xy_data


def get_data_loader(args):
    """
    creates tensorflow data iterators for train and test data (which it loads) along with corresponding
    loading functions and dictionaries and packs them into 'data_loader' tuples
    :param args: run arguments
    :return:
    batch data variable tuple: variables to use like placeholders in models
    batch indexing variable : used to identify correct Z vals to load for batch if Zs are stored
    data_loaders: tuple of dataset iterator and necesary loading ops
    """

    train_xy_data, test_xy_data = get_datasets(args)

    n_feats = train_xy_data.x.shape[1]
    n_samples_train = train_xy_data.x.shape[0]

    data_x = tf.get_variable(name='x_data', shape=(n_samples_train, n_feats))
    data_x_pl = tf.placeholder(tf.float32, shape=(None, n_feats))

    train_dict = {data_x_pl: train_xy_data.x}
    test_dict = {data_x_pl: test_xy_data.x}

    dataset_pls = [data_x_pl]
    feed_sampler_ops = [tf.assign(data_x, data_x_pl)]

    if train_xy_data.y is not None:
        n_labels = train_xy_data.y.shape[1]
        data_y = tf.get_variable(name='y_data', shape=(n_samples_train, n_labels))
        data_y_pl = tf.placeholder(tf.float32, shape=(None, n_labels))
        train_dict[data_y_pl] = train_xy_data.y
        test_dict[data_y_pl] = test_xy_data.y
        dataset_pls.append(data_y_pl)
        feed_sampler_ops.append(tf.assign(data_y, data_y_pl))

    feed_sampler_op = tf.group(feed_sampler_ops)

    dataset_tf = tf.data.Dataset.from_tensor_slices(tuple(dataset_pls))
    data_iter, feed_iter_op, xy_batch_var = make_iter_and_load_op(dataset_tf, train_xy_data, args)

    loader = data_loader(feed_sampler_op, data_iter, feed_iter_op, train_dict, test_dict)

    return xy_batch_var, loader


def make_iter_and_load_op(dataset, train_xy_data, args):
    n_feats = train_xy_data.x.shape[1]

    dataset = dataset.batch(args.batch_size)
    data_iter = dataset.make_initializable_iterator()
    iter_outputs = data_iter.get_next()
    x_iter = iter_outputs[0]

    x_in = tf.get_variable(name='x_in', shape=(args.batch_size, n_feats), dtype=tf.float32)
    load_batch_ops = [tf.assign(x_in, x_iter)]

    if len(iter_outputs) > 1:
        n_labels = train_xy_data.y.shape[1]
        y_in = tf.get_variable(name='y_in', shape=(args.batch_size, n_labels), dtype=tf.float32)
        y_iter = iter_outputs[1]
        load_batch_ops.append(tf.assign(y_in, y_iter))
    else:
        y_in = None
    load_batch_op = tf.group(load_batch_ops)

    xy_batch_var = xy_data(x_in, y_in)
    return data_iter, load_batch_op, xy_batch_var
