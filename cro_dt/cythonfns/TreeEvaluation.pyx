import numpy as np
cimport numpy as np
cimport cython

np.import_array()
@cython.wraparound(False)
@cython.boundscheck(False)

# Given the observation "x", returns the leaf index in which "x" falls in the decision tree W.
# "used_attributes" is a vector of size "W.shape[0] - 1" that contains the index of the attribute
# used to split the node. For example, if "used_attributes[0] = 2", then the first node in the tree
# is split using the attribute "x_2"
cdef get_leaf_index(np.ndarray[np.double_t, ndim=1] X,
                    np.ndarray[np.double_t, ndim=2] W,
                    np.ndarray[Py_ssize_t, ndim=1] attributes,
                    np.ndarray[np.double_t, ndim=1] thresholds,
                    np.ndarray[np.int64_t, ndim=1] inversions,
                    int depth):

    cdef Py_ssize_t node_idx = 0
    cdef Py_ssize_t leaf_idx = 0
    cdef int curr_depth = depth - 1
    cdef double total
    cdef int P = X.shape[0]

    while curr_depth >= 0:
        if inversions[node_idx] * X[attributes[node_idx]] <= thresholds[node_idx]:
            node_idx += 1
        else:
            node_idx += 2 ** curr_depth
            leaf_idx += 2 ** curr_depth
        curr_depth -= 1

    return leaf_idx

# Same as "sum(np.max(np.array(count), axis=1)) / X.shape[0]", re-written in Cython
cdef get_accuracy(np.ndarray[np.double_t, ndim=2] count, int n_samples):
    cdef int n_leaves = count.shape[0]
    cdef int n_classes = count.shape[1]
    cdef double max_count
    cdef double accuracy = 0.0

    cdef Py_ssize_t i, j

    for i in range(n_leaves):
        max_count = count[i][0]
        for j in range(1, n_classes):
            if count[i][j] > max_count:
                max_count = count[i][j]
        accuracy += max_count
    accuracy /= n_samples

    return accuracy

# Same as "np.argmax(count, axis=1)", re-written in Cython
cdef argmax_loop(np.ndarray[np.double_t, ndim=2] arr):
    cdef int n_rows = arr.shape[0]
    cdef int n_cols = arr.shape[1]
    cdef np.ndarray[np.int64_t, ndim=1] indices = np.zeros(n_rows, dtype=np.int64)

    cdef Py_ssize_t i, j

    for i in range(n_rows):
        max_val = arr[i, 0]
        max_idx = 0
        for j in range(1, n_cols):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
                max_idx = j
        indices[i] = max_idx
    return indices

# Calculate the accuracy and labels of the decision tree W, represented as a matrix.
# Current code is optimized for univariate trees and does not work for multivariate ones.
def dt_tree_fit(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.int64_t, ndim=1] y,
                np.ndarray[np.double_t, ndim=2] W, int depth, int n_classes,
                np.ndarray[np.double_t, ndim=2] X_=None, Y_=None, M=None):

    cdef int n_leaves
    cdef double accuracy
    cdef np.ndarray[np.double_t, ndim=2] count
    cdef np.ndarray[np.int64_t, ndim=1] labels
    cdef int N = X_.shape[0]
    cdef int P = W.shape[0]
    cdef Py_ssize_t i, j

    n_leaves = W.shape[0] + 1
    count = np.zeros((n_leaves, n_classes))

    attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
    thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
    inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)

    # Assigns each observation to a leaf and keeps count of the classes in each leaf
    for i in range(N):
        leaf = get_leaf_index(X_[i], W, attributes, thresholds, inversions, depth)
        count[leaf][y[i]] += 1

    accuracy = sum(np.max(np.array(count), axis=1)) / X.shape[0]
    labels = np.argmax(count, axis=1)
    # accuracy = get_accuracy(count, N)
    # labels = argmax_loop(count)

    return accuracy, labels