import numpy as np
from scipy.linalg import sqrtm

def cal_w_distance(f1, f2):
    miu_f1 = np.mean(f1, axis=0)
    miu_f2 = np.mean(f2, axis=0)
    cov_f1 = np.cov(f1.T)
    cov_f2 = np.cov(f2.T)

    # Add a small regularization term to the diagonals of the covariance matrices
    cov_f1 += np.eye(cov_f1.shape[0]) * 1e-6
    cov_f2 += np.eye(cov_f2.shape[0]) * 1e-6

    sqrt_cov_product = sqrtm(cov_f1.dot(cov_f2))

    # Ensure the result is real
    if np.iscomplexobj(sqrt_cov_product):
        sqrt_cov_product = np.real(sqrt_cov_product)

    delta_miu = miu_f1 - miu_f2
    w_d = np.sum(delta_miu**2) + cov_f1.trace() + cov_f2.trace() - \
        2 * sqrt_cov_product.trace()

    return np.sqrt(abs(w_d))


def cal_variety(matrix):
    eps = 1e-3
    row_norms = np.sum(matrix * matrix, axis=1)
    pairwise_inner_products = np.dot(matrix, matrix.T)
    pairwise_distances_squared = np.expand_dims(
        row_norms, axis=1) + np.expand_dims(row_norms, axis=0) - 2 * pairwise_inner_products
    pairwise_distances = 1 / np.sqrt(pairwise_distances_squared+eps)
    return np.sum(np.triu(pairwise_distances, k=1)) / len(pairwise_distances) / (len(pairwise_distances)-1) * 2