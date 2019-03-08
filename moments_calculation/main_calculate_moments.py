# __author__ = 'mijung'

from builtins import range

import numpy as np
import moments_calculation.calculate_moments_accountant as ma


""" define variables for your training setting """
default_n_data = 60000  # number of training datapoints
default_n_epochs = 20  # number of epochs
default_batch_size = 100  # mini-batch size
default_total_del = 1e-5  # accepted delta


def calculate_epsilon(n_data=60000, n_epochs=20, batch_size=50, sigma=1., total_del=1e-5, verify=True):
    """ ========  calculate  moments  ======== """

    steps = n_data / batch_size * n_epochs  # number of iterations
    q = (batch_size / n_data)  # sampling rate

    # print(steps, q)
    # make sure lambda < sigma^2 log (1/(nu*sigma))
    max_lmbd = int(np.floor((sigma ** 2) * np.log(1 / (q * sigma))))
    max_lmbd *= 2
    # print(max_lmbd)

    lmbds = range(1, max_lmbd + 1)
    log_moments = []
    for lmbd in lmbds:
        # log_moment = 0
        log_moment = ma.compute_log_moment(q, sigma, steps, lmbd, verify=verify, verbose=False)
        # print(log_moment)
        log_moments.append((lmbd, log_moment))

    total_epsilon, total_delta = ma.get_privacy_spent(log_moments, target_delta=total_del)

    print("total privacy loss computed by moments accountant is {}".format(total_epsilon))

    return total_epsilon, total_delta


def dp_pca_addition(sig, delta):
    eps = np.sqrt(2 * np.log(5 / (4 * delta))) / sig
    return eps


def main():
    n_data = 60000
    epsilons = []
    sigmas = [5.3, 5.]
    epochs = [30]
    batch_size = 500
    ma_delta = 1e-5
    pca_sigma = None
    pca_delta = 1e-6
    pca_eps = dp_pca_addition(pca_sigma, pca_delta) if pca_sigma else None
    print('pca epsilon:', pca_eps)
    for s in sigmas:
        for ep in epochs:
            eps, delta = calculate_epsilon(n_data, ep, batch_size, s, total_del=ma_delta, verify=False)
            epsilons.append(eps if pca_eps is None else eps + pca_eps)
            print('sigma:', s, 'delta:', ma_delta + pca_delta)
            print('n_epochs: ', ep)
            print('')
            print('-----------------------------------------------------------------------------------------------')
    print('epsilons', epsilons)
    eps_diffs = [epsilons[i+1] - epsilons[i] for i in range(len(epsilons) - 1)]
    print('diffs', eps_diffs)
    print('diff diffs', [e - eps_diffs[0] for e in eps_diffs])


if __name__ == '__main__':
    main()
