# __author__ = 'frederik harder'
import os
import matplotlib.pyplot as plt
import numpy as np


def read_errors(file_path):
    errs = np.load(file_path)
    max_err = np.max(errs)
    last_err = errs[-1]
    return errs, max_err, last_err


def ae_convergence_plot(shared_prefix, suffixes, modes, plot_file,
                        start_dir='../../results/', te_err_file='test_errors.npy', tr_err_file='train_errors.npy'):

    def make_data_mat(exp_prefix, err_file):
        exp_dirs = [k for k in os.listdir(start_dir) if k.startswith(exp_prefix)]
        err_paths = [os.path.join(start_dir, k, err_file) for k in exp_dirs]
        read_errs = [read_errors(k)[0] for k in err_paths]
        vec_errs = [np.asarray(k) for k in read_errs]
        err_block = np.stack(vec_errs, axis=1)
        return err_block

    means_test = []
    means_train = []
    stds_test = []
    stds_train = []
    for cat in suffixes:
        eps_test = make_data_mat(shared_prefix + cat, te_err_file)
        eps_train = make_data_mat(shared_prefix + cat, tr_err_file)
        means_test.append(np.mean(eps_test, axis=1))
        stds_test.append(eps_test.std(axis=1))
        means_train.append(np.mean(eps_train, axis=1))
        stds_train.append(eps_train.std(axis=1))

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(8, 6))
    plt.title("DP-MAC Autoencoder")

    min_y = 2
    max_y = 18

    plt.xlabel("epoch")
    plt.xlim(1, 50)
    plt.xticks(np.linspace(0, 50, 11))

    plt.ylabel("MSE")
    plt.ylim(min_y, max_y)
    plt.yticks(np.linspace(min_y, max_y, 9))
    plt.hlines(list(range(min_y, max_y, 1)), 0, 50, colors='grey', linestyles='dashed', linewidths=0.5)

    colors = dict(zip(modes, ["y", "c", "b", "r", "g", 'm', 'k', '#888888']))
    for exp, err_test, std_test, err_train, std_train in zip(modes, means_test, stds_test, means_train, stds_train):
        err_test = err_test.flatten()
        std_test = std_test.flatten()
        err_train = err_train.flatten()
        # std_train = std_train.flatten()
        print(err_test)
        plt.plot(1 + np.arange(len(err_test)), err_test, label="{} (test)".format(exp), c=colors[exp])
        plt.plot(1 + np.arange(len(err_train)), err_train, "x", label="{} (train)".format(exp), c=colors[exp])
        plt.fill_between(1 + np.arange(len(err_test)), err_test - std_test, err_test + std_test, alpha=0.1,
                         color=colors[exp])

    plt.legend(loc='upper right', ncol=2, frameon=False, fontsize=10)
    plt.savefig(plot_file)


def cl_convergence_plot(shared_prefix, suffixes, modes, plot_file,
                        start_dir='../../results/', te_err_file='test_errors.npy', tr_err_file='train_errors.npy'):

    def make_data_mat(exp_prefix, err_file):
        exp_dirs = [k for k in os.listdir(start_dir) if k.startswith(exp_prefix)]
        err_paths = [os.path.join(start_dir, k, err_file) for k in exp_dirs]
        read_errs = [read_errors(k)[0] for k in err_paths]
        vec_errs = [np.asarray(k) for k in read_errs]
        err_block = np.stack(vec_errs, axis=1)
        return err_block

    means_test = []
    means_train = []
    stds_test = []
    stds_train = []
    for cat in suffixes:
        eps_test = make_data_mat(shared_prefix + cat, te_err_file)
        eps_train = make_data_mat(shared_prefix + cat, tr_err_file)
        # print(eps_test)
        # print(eps_train)
        means_test.append(np.median(eps_test, axis=1))
        stds_test.append(eps_test.std(axis=1))
        means_train.append(np.median(eps_train, axis=1))
        stds_train.append(eps_train.std(axis=1))

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(8, 6))
    plt.title("DP-MAC Classifier")

    n_ep = 30
    min_acc = 0.84
    max_acc = 1.
    x_ticks = 7
    y_ticks = 9

    plt.xlabel("epoch")
    plt.xlim(1, n_ep)
    plt.xticks(np.linspace(0, n_ep, x_ticks))

    plt.ylabel("Accuracy")
    plt.ylim(min_acc, max_acc)
    plt.yticks(np.linspace(min_acc, max_acc, y_ticks))
    plt.hlines(np.linspace(min_acc, max_acc, y_ticks), 1, n_ep, colors='grey', linestyles='dashed', linewidths=0.5)

    colors = dict(zip(modes, ["y", "c", "b", "r", "g", 'm', 'k', '#888888']))
    for exp, err_test, std_test, err_train, std_train in zip(modes, means_test, stds_test, means_train, stds_train):
        err_test = err_test.flatten()
        std_test = std_test.flatten()
        err_train = err_train.flatten()
        # std_train = std_train.flatten()
        print(err_test)

        plt.plot(1 + np.arange(len(err_test)), err_test, label="{} (test)".format(exp), c=colors[exp])
        plt.plot(1 + np.arange(len(err_train)), err_train, "x", label="{} (train)".format(exp), c=colors[exp])
        plt.fill_between(1 + np.arange(len(err_test)), err_test - std_test, err_test + std_test, alpha=0.1,
                         color=colors[exp])

    plt.legend(loc='lower right', ncol=2, frameon=False, fontsize=10)
    plt.savefig(plot_file)


def sgd_ae_convergence_plot(shared_prefix, suffixes, modes, plot_file,
                            start_dir='../../results/', tr_err_file='train_log.npy', te_err_file='test_log.npy'):

    def make_data_mat(exp_prefix, err_file):
        print([k for k in os.listdir(start_dir) if k.startswith('grid')], exp_prefix)
        exp_dirs = [k for k in os.listdir(start_dir) if k.startswith(exp_prefix)]
        err_paths = [os.path.join(start_dir, k, err_file) for k in exp_dirs]
        print(err_paths)
        read_errs = [read_errors(k)[0] for k in err_paths]
        vec_errs = [np.asarray(k) for k in read_errs]
        err_block = np.stack(vec_errs, axis=1)
        return err_block

    means_test = []
    means_train = []
    stds_test = []
    stds_train = []
    for cat in suffixes:
        eps_test = make_data_mat(shared_prefix + cat, te_err_file)
        eps_train = make_data_mat(shared_prefix + cat, tr_err_file)
        means_test.append(np.mean(eps_test, axis=1))
        stds_test.append(eps_test.std(axis=1))
        means_train.append(np.mean(eps_train, axis=1))
        stds_train.append(eps_train.std(axis=1))

    min_y = 2
    max_y = 18
    max_x = 100

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(8, 6))
    plt.title("DP-SGD Autoencoder")

    plt.xlabel("epoch")
    plt.xlim(0, max_x)
    plt.xticks(np.linspace(0, max_x, 11))

    plt.ylabel("MSE")
    plt.ylim(min_y, max_y)
    plt.yticks(np.linspace(min_y, max_y, 9))
    plt.hlines(list(range(min_y, max_y, 1)), 0, max_x, colors='grey', linestyles='dashed', linewidths=0.5)

    colors = dict(zip(modes, ["y", "c", "b", "r", "g", 'm', 'k', '#888888']))
    for exp, err_test, std_test, err_train, std_train in zip(modes, means_test, stds_test, means_train, stds_train):
        err_test = err_test.flatten()
        std_test = std_test.flatten()
        err_train = err_train.flatten()
        # std_train = std_train.flatten()
        print(err_test)
        plt.plot(1 + np.arange(len(err_test)), err_test, label="{} (test)".format(exp), c=colors[exp])
        plt.plot([1 + 2 * k for k in np.arange(len(err_train[::2]))], err_train[::2], "x",
                 label="{} (train)".format(exp), c=colors[exp])
        plt.fill_between(1 + np.arange(len(err_test)), err_test - std_test, err_test + std_test, alpha=0.1,
                         color=colors[exp])

    plt.legend(loc='upper right', ncol=2, frameon=False, fontsize=10)
    plt.savefig(plot_file)


if __name__ == '__main__':
    # cl_convergence_plot(shared_prefix='run_v',
    #                     suffixes=('0', '1', '4', '3'),
    #                     modes=['$\epsilon=8$', '$\epsilon=2$', '$\epsilon=0.5$', 'NP'],
    #                     plot_file='dp-mac-cl-convergence.png',
    #                     start_dir='../../results/gridsearch_cl_all_eps_v1/',
    #                     tr_err_file='train_errors.npy', te_err_file='test_errors.npy')

    # ae_convergence_plot('run_v',
    #                     ['3', '2', '1', '0', '4'],
    #                     ['$\epsilon=1$', '$\epsilon=2$', '$\epsilon=4$', '$\epsilon=8$', 'NP'],
    #                     'dp-mac-ae-convergence.png',
    #                     start_dir='../../results/gridsearch_ae_all_eps_v1/',
    #                     tr_err_file='train_errors.npy', te_err_file='test_errors.npy')

    sgd_ae_convergence_plot('run_v',
                            ['3', '2', '1', '0', '4'],
                            ['$\epsilon=1$', '$\epsilon=2$',  '$\epsilon=4$', '$\epsilon=8$', 'NP'],
                            'dp-sgd-ae-convergence.png',
                            start_dir='../../results/gridsearch_dpsgd_ae_v1/',
                            tr_err_file='train_log.npy', te_err_file='test_log.npy')
