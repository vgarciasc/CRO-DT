import numpy as np
import cro_dt.VectorTree as vt

def dt_matrix_fit(X, y, W, depth, n_classes, X_=None, Y_=None, M=None):
    n_leaves = len(W) + 1
    N = len(X)

    M = vt.create_mask_dx(depth) if M is None else M
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T if X_ is None else X_
    Y_ = np.tile(y, (n_leaves, 1)) if Y_ is None else Y_

    Z = np.sign(W @ X_.T)
    Z_ = np.clip(M @ Z - (depth - 1), 0, 1)
    R = Z_ * Y_

    count_0s = N - np.sum(Z_, axis=1)
    R = np.int_(R)
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