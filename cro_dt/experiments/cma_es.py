import argparse
import copy
from datetime import datetime
from cro_dt.sup_configs import get_config, load_dataset
import time
import cma
import numpy as np
from cro_dt.SoftTree import SoftTree
import cro_dt.TensorflowTree as tft
import cro_dt.VectorTree as vt
from sklearn.datasets import make_blobs
import tensorflow as tf

import matplotlib.pyplot as plt
from rich import print


def plot_decision_surface(X, y, model):
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    grid = np.hstack((r1, r2))
    yhat = np.array([model.predict(row) for row in grid])

    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')

    for class_value in range(2):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

    plt.show()

def get_accuracy(weights, X, y):
    tree.weights = weights.reshape((tree.num_nodes, tree.num_attributes + 1))

    tree.update_leaves_by_dataset(X, y)
    y_pred = tree.predict_batch(X)
    accuracy = np.mean([(1 if y_pred[i] == y[i] else 0) for i in range(len(X))])

    return 1 - accuracy

def get_accuracy_tf(weights, X, y, depth, n_classes, X_, Y_, M, N, n_leaves):
    W = weights.reshape((n_leaves-1, -1))
    Wt = tf.convert_to_tensor(W, dtype=tf.float64)

    acc, _ = tft.dt_matrix_fit_nb(X, y, Wt, depth, n_classes, X_, Y_, M, N, n_leaves)

    return 1 - float(acc)

if __name__ == "__main__":
    depth = 2
    n_classes = 2
    n_leaves = 2 ** depth
    M = vt.create_mask_dx(depth)
    N = 1000

    tree = SoftTree(num_attributes=2, num_classes=2)
    tree.randomize(depth=depth)

    X, y = make_blobs(n_samples=N, centers=[[-1, 1], [1, 1], [1, -1], [-1, -1]], n_features=2, random_state=1, cluster_std=0.5)
    y = np.array([y_i % 2 for y_i in y])

    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.int_(np.tile(y, (n_leaves, 1)))

    Xt = tf.convert_to_tensor(X, dtype=tf.float64)
    X_t = tf.convert_to_tensor(X_, dtype=tf.float64)
    Y_t = tf.convert_to_tensor(Y_ + 1, dtype=tf.int32)
    Mt = tf.convert_to_tensor(M, dtype=tf.int32)

    accuracy = get_accuracy(tree.weights, X, y)
    print(f"Accuracy: {accuracy}")

    x0 = tree.weights.flatten()
    sigma0 = 1

    x, es = cma.fmin2(get_accuracy_tf, x0, sigma0,
                      options={'maxfevals': 10e4,
                               'bounds': [-1, 1]},
                      args=(Xt, y, depth, n_classes, X_t, Y_t, Mt, N, n_leaves))

    tree.weights = x.reshape((tree.num_nodes, tree.num_attributes + 1))
    tree.update_leaves_by_dataset(X, y)
    print(tree)
    print(f"Accuracy: {1 - get_accuracy(tree.weights, X, y)}")

    plot_decision_surface(X, y, tree)