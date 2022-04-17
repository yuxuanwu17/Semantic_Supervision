from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import torch
import numpy as np


def rbf_kernel_np(X, Y=None, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    K = np.exp(K)  # exponentiate K in-place
    return K


# Pairwise distances
def euclidean_distances(X, Y=None, squared=False):
    return _euclidean_distances(X, Y, squared)


def _euclidean_distances(X, Y, squared=False):
    XX = row_norms(X, squared=True)[:, np.newaxis]
    YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = -2 * np.dot(X, Y.T)
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)


def row_norms(X, squared=False):
    norms = np.einsum("ij,ij->i", X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms


if __name__ == '__main__':
    np.random.seed(12)
    input_rep = np.random.rand(64, 512)
    label_rep = np.random.rand(512, 100)
    logits_selfImpl = rbf_kernel_np(input_rep, label_rep.T)
    print("==================")
    print(logits_selfImpl)
    print(pairwise_kernels(input_rep, label_rep.T, metric='rbf'))
    print("==================")



