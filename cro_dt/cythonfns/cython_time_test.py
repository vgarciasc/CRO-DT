import pdb
import numpy as np
import time
from cro_dt.VectorTree import dt_matrix_fit_dx
import cro_dt.VectorTree as vt
import cro_dt.cythonfns.TreeEvaluation as cy
import cro_dt.NumbaTree as nt
# import cro_dt.CupyTree as ct
import tensorflow as tf
import cro_dt.TensorflowTree as tft

if __name__ == "__main__":
    depth = 6

    n_samples = 10000
    n_classes = 2
    n_leaves = 2 ** depth
    n_features = 2

    simulations = 100

    X = np.random.uniform(-5, 5, (n_samples, n_features))
    X = np.int_(X).astype(np.double)
    y = np.random.randint(0, n_classes, n_samples)
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.tile(y, (2 ** depth, 1))
    M = vt.create_mask_dx(depth)

    tic = time.perf_counter()
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)

        acc_tree, _ = vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_)
    toc = time.perf_counter()
    print(f"Tree evaluation time: \t\t\t{(toc - tic)} s")

    tic = time.perf_counter()
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)
        acc_cytree, _ = cy.dt_tree_fit(X, y, W, depth, n_classes, X_)
    toc = time.perf_counter()
    print(f"Cytree evaluation time: \t\t{(toc - tic)} s")

    tic = time.perf_counter()
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)
        acc_matrix, _ = dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
    toc = time.perf_counter()
    print(f"Matrix evaluation time: \t\t{(toc - tic)} s")

    tic = time.perf_counter()
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)
        with tf.device("/GPU:0"):
            X = tf.convert_to_tensor(X, dtype=tf.float64)
            W = tf.convert_to_tensor(W, dtype=tf.float64)
            X_ = tf.convert_to_tensor(X_, dtype=tf.float64)
            Y_ = tf.convert_to_tensor(Y_, dtype=tf.int32)
            M = tf.convert_to_tensor(M, dtype=tf.int32)

            acc_tensorflow, _ = tft.dt_matrix_fit(X, None, W, depth, n_classes, X_, Y_, M)
    toc = time.perf_counter()
    print(f"Tensorflow evaluation time: \t{(toc - tic)} s")
