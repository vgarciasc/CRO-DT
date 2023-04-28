import numpy as np


def get_count(X_, y, W, depth, n_classes, attributes, thresholds, inversions):
    count = np.zeros((W.shape[0] + 1, n_classes), dtype=np.int64)

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


def dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions):
    count = np.zeros((W.shape[0] + 1, n_classes), dtype=np.int64)
    count = get_count(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
    return sum(np.max(np.array(count), axis=1)) / X_.shape[0], np.argmax(count, axis=1)
