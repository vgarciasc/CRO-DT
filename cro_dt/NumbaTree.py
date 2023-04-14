import pdb
import numpy as np
from functools import reduce
from numba import jit
from numba import cuda
import time
from rich import print
import cro_dt.VectorTree as vt

def dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M):
    n_leaves = len(W) + 1
    N = len(X)

    Z = np.sign(W @ X_.T)
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    count_0s = N - np.sum(Z_, axis=1)
    R = np.int_(R)
    H = np.zeros((n_leaves, n_classes))
    labels = np.zeros(n_leaves)
    correct_preds = 0
    for l in range(n_leaves):
        bc = np.bincount(R[l], minlength=n_classes)
        bc[0] -= count_0s[l]

        most_popular_class = np.argmax(bc)
        labels[l] = most_popular_class
        correct_preds += bc[most_popular_class]

    accuracy = correct_preds / N
    return accuracy, labels

def dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_=None, Y_=None, M=None, untie=False):
    n_leaves = len(W) + 1
    N = len(X)

    M = vt.create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_
    S = np.ones(N)

    R, Z_ = numba_subproc1(X_, Y_, M, S, W, depth, n_classes)

    bc = [np.ones_like(R) for _ in range(0, n_classes)]
    for i in range(0, n_classes):
        for j in range(0, n_classes):
            if j != i:
                bc[i] *= (j - R) / (j - i)

    H = np.stack(bc) @ np.ones(N)
    accuracy, labels = numba_subproc2(H, Z_, S, N, n_leaves)
    return accuracy, labels

@jit(nopython=True)
def numba_subproc1(X_, Y_, M, S, W, depth, n_classes):
    Z = np.sign(W @ X_.T)
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    return R, Z_

@jit(nopython=True)
def numba_subproc2(H, Z_, S, N, n_leaves):
    H[0] -= (np.ones(n_leaves) * N - Z_ @ S)  # removing false zeros

    labels = [np.argmax(r) for r in H.T]
    accuracy = sum([np.max(r) for r in H.T]) / N

    return accuracy, labels