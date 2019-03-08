import numpy as np


def private_pca(x_data, sigma, pca_dims):
    x_data = np.float32(x_data)
    n_samples = x_data.shape[0]
    x_dims = x_data.shape[1]

    x_data_norms = np.linalg.norm(x_data, 2, axis=1, keepdims=True)
    x_data_normed = x_data / x_data_norms
    cov_acc = x_data_normed.T @ x_data_normed

    if sigma is not None and sigma > 0.:
        noise_mat = np.random.normal(size=(x_dims, x_dims), scale=sigma)
        i_lower = np.tril_indices(x_dims, -1)
        noise_mat[i_lower] = noise_mat.T[i_lower]  # symmetric noise
        cov_acc += noise_mat

    dp_cov = cov_acc / n_samples
    s, _, v = np.linalg.svd(dp_cov)
    sing_vecs = v.T[:, :pca_dims]

    return sing_vecs


def pca_preprocess(x_data_train, x_data_test, args):

    x_data_train = x_data_train - x_data_train.mean()
    x_data_train = x_data_train / np.abs(x_data_train).max()
    x_data_test = x_data_test - x_data_test.mean()
    x_data_test = x_data_test / np.abs(x_data_test).max()

    sing_vecs = private_pca(x_data_train, args.dp_pca_sigma, args.dp_pca_dims)

    x_train_pca = x_data_train @ sing_vecs  # (n,dx) (dimx, dimx') -> (bs,dimx')
    x_test_pca = x_data_test @ sing_vecs

    return x_train_pca, x_test_pca
