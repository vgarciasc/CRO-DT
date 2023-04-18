import numpy as np

# Given the observation "x", returns the leaf index in which "x" falls in the decision tree W.
# "used_attributes" is a vector of size "W.shape[0] - 1" that contains the index of the attribute
# used to split the node. For example, if "used_attributes[0] = 2", then the first node in the tree
# is split using the attribute "x_2"

def get_leaf_index(X, W, attributes, thresholds, inversions, depth):
    node_idx = 0
    leaf_idx = 0
    curr_depth = depth - 1

    while curr_depth >= 0:
        if inversions[node_idx] * X[attributes[node_idx]] <= thresholds[node_idx]:
            node_idx += 1
        else:
            node_idx += 2 ** curr_depth
            leaf_idx += 2 ** curr_depth
        curr_depth -= 1

    return leaf_idx

# Calculate the accuracy and labels of the decision tree W, represented as a matrix.
# Current code is optimized for univariate trees and does not work for multivariate ones.
def dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions):

    N = X_.shape[0]
    P = W.shape[0]

    n_leaves = W.shape[0] + 1
    count = np.zeros((n_leaves, n_classes))

    # Assigns each observation to a leaf and keeps count of the classes in each leaf
    for i in range(N):
        leaf = get_leaf_index(X_[i], W, attributes, thresholds, inversions, depth)
        count[leaf][y[i]] += 1

    accuracy = sum(np.max(np.array(count), axis=1)) / N
    labels = np.argmax(count, axis=1)
    # accuracy = get_accuracy(count, N)
    # labels = argmax_loop(count)

    return accuracy, labels