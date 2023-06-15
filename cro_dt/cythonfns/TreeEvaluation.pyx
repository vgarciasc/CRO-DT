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

cdef int get_leaf_index(np.ndarray[np.double_t, ndim=1] X,
                    np.ndarray[Py_ssize_t, ndim=1] attributes,
                    np.ndarray[np.double_t, ndim=1] thresholds,
                    np.ndarray[np.int64_t, ndim=1] inversions,
                    int depth):

    cdef Py_ssize_t node_idx = 0
    cdef Py_ssize_t leaf_idx = 0
    cdef int curr_depth = depth - 1
    cdef int i = 0
    cdef int j

    while curr_depth >= 0:
        i += 1
        if inversions[node_idx] * X[attributes[node_idx]] <= thresholds[node_idx]:
            node_idx += 1
        else:
            j = 2 ** curr_depth
            node_idx += j
            leaf_idx += j
        curr_depth -= 1

    return leaf_idx

def get_leaf_simple(np.ndarray[np.double_t, ndim=1] x,
                    np.ndarray[Py_ssize_t, ndim=1] attributes,
                    np.ndarray[np.double_t, ndim=1] thresholds,
                    int depth):

    cdef Py_ssize_t curr_depth = depth - 1
    cdef Py_ssize_t node_idx = 0
    cdef Py_ssize_t leaf_idx = 0

    while curr_depth >= 0:
        if x[attributes[node_idx]] <= thresholds[node_idx]:
            node_idx += 1
        else:
            node_idx += 2 ** curr_depth
            leaf_idx += 2 ** curr_depth
        curr_depth -= 1

    return leaf_idx

# # Same as "sum(np.max(np.array(count), axis=1)) / X.shape[0]", re-written in Cython
# cdef get_accuracy(np.ndarray[np.double_t, ndim=2] count, int n_samples):
#     cdef int n_leaves = count.shape[0]
#     cdef int n_classes = count.shape[1]
#     cdef double max_count
#     cdef double accuracy = 0.0
#
#     cdef Py_ssize_t i, j
#
#     for i in range(n_leaves):
#         max_count = count[i][0]
#         for j in range(1, n_classes):
#             if count[i][j] > max_count:
#                 max_count = count[i][j]
#         accuracy += max_count
#     accuracy /= n_samples
#
#     return accuracy
#
# # Same as "np.argmax(count, axis=1)", re-written in Cython
# cdef argmax_loop(np.ndarray[np.double_t, ndim=2] arr):
#     cdef int n_rows = arr.shape[0]
#     cdef int n_cols = arr.shape[1]
#     cdef np.ndarray[np.int64_t, ndim=1] indices = np.zeros(n_rows, dtype=np.int64)
#
#     cdef Py_ssize_t i, j
#
#     for i in range(n_rows):
#         max_val = arr[i, 0]
#         max_idx = 0
#         for j in range(1, n_cols):
#             if arr[i, j] > max_val:
#                 max_val = arr[i, j]
#                 max_idx = j
#         indices[i] = max_idx
#     return indices

# Calculate the accuracy and labels of the decision tree W, represented as a matrix.
# Current code is optimized for univariate trees and does not work for multivariate ones.
cdef np.ndarray[Py_ssize_t, ndim=2] get_count(np.ndarray[np.double_t, ndim=2] X_,
                                              np.ndarray[np.int64_t, ndim=1] y,
                                              np.ndarray[np.double_t, ndim=2] W,
                                              int depth, int n_classes,
                                              np.ndarray[Py_ssize_t, ndim=1] attributes,
                                              np.ndarray[np.double_t, ndim=1] thresholds,
                                              np.ndarray[np.int64_t, ndim=1] inversions):

    cdef np.ndarray[Py_ssize_t, ndim=2] count = np.zeros((W.shape[0] + 1, n_classes), dtype=np.int64)
    cdef np.ndarray[np.double_t, ndim=1] X
    cdef Py_ssize_t node_idx
    cdef Py_ssize_t leaf_idx
    cdef int curr_depth
    cdef int i
    cdef int j

    # Assigns each observation to a leaf and keeps count of the classes in each leaf
    for i in range(X_.shape[0]):

        node_idx = 0
        leaf_idx = 0
        curr_depth = depth - 1
        X = X_[i]

        while curr_depth >= 0:
            if inversions[node_idx] * X[attributes[node_idx]] <= thresholds[node_idx]:
                node_idx += 1
            else:
                j = 2 ** curr_depth
                node_idx += j
                leaf_idx += j
            curr_depth -= 1

        count[leaf_idx][y[i]] += 1

    return count

cpdef dt_tree_fit(np.ndarray[np.double_t, ndim=2] X_, np.ndarray[np.int64_t, ndim=1] y,
                np.ndarray[np.double_t, ndim=2] W, int depth, int n_classes,
                np.ndarray[Py_ssize_t, ndim=1] attributes,
                np.ndarray[np.double_t, ndim=1] thresholds,
                np.ndarray[np.int64_t, ndim=1] inversions):

    cdef np.ndarray[Py_ssize_t, ndim=2] count = np.zeros((W.shape[0] + 1, n_classes), dtype=np.int64)
    count = get_count(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
    return sum(np.max(np.array(count), axis=1)) / X_.shape[0], np.argmax(count, axis=1)

    # No performance difference between using functions above and explicitly implementing the below

    # cdef np.ndarray[Py_ssize_t, ndim=1] labels = np.zeros(W.shape[0] + 1, dtype=np.int64)
    # cdef Py_ssize_t leaf_idx, label, max_label, max_label_pop
    # cdef Py_ssize_t correct_predictions = 0

    # for leaf_idx, leaf_count in enumerate(count):
    #     max_label = 0
    #
    #     for label_idx, label_count in enumerate(leaf_count):
    #         if label_count > max_label:
    #             max_label = label_idx
    #             max_label_pop = label_count
    #
    #     labels[leaf_idx] = max_label
    #     correct_predictions += max_label_pop
    #
    # return correct_predictions / X_.shape[0], labels

def dt_matrix_fit(np.ndarray[np.double_t, ndim=2] X_,
                  np.ndarray[np.int64_t, ndim=2] Y_,
                  np.ndarray[np.double_t, ndim=2] W,
                  np.ndarray[np.int64_t, ndim=2] M,
                  int depth, int n_classes):
    cdef int N = len(X_)

    cdef np.ndarray[np.double_t, ndim=2] Z = np.clip(M @ np.sign(W @ X_.T) - (depth - 1), 0, 1)
    cdef np.ndarray[np.int64_t, ndim=2] BC = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=np.int_(Z * Y_))
    BC[:, 0] -= np.subtract(N, np.sum(Z, axis=1, dtype=np.int64))
    cdef double accuracy = np.sum(np.max(BC, axis=1)) / N
    cdef np.ndarray[np.int64_t, ndim=1] labels = np.argmax(BC, axis=1)

    return accuracy, labels