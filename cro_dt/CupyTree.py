import numpy as np
import cro_dt.VectorTree as vt
import cupy as cp

def dt_matrix_fit_subproc1(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    X_ = cp.asarray(X_)
    Y_ = cp.asarray(Y_)
    W = cp.asarray(W)
    M = cp.asarray(M)
    N = len(X)

    Z = cp.sign(W @ X_.T)
    Z_ = cp.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    count_0s = N - cp.sum(Z_, axis=1)
    R = cp.int_(R.get())
    return R, count_0s

def dt_matrix_fit(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    R, count_0s = dt_matrix_fit_subproc1(X, y, W, depth, n_classes, X_, Y_, M)
    n_leaves = len(W) + 1
    N = len(X)

    labels = cp.zeros(n_leaves)
    correct_preds = 0
    for l in range(n_leaves):
        bc = np.bincount(R[l], minlength=n_classes)
        bc[0] -= count_0s[l]

        most_popular_class = np.argmax(bc)
        labels[l] = most_popular_class
        correct_preds += bc[most_popular_class]

    accuracy = correct_preds / N
    return accuracy, labels