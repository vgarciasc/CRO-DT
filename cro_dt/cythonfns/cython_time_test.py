import pdb
import numpy as np
import time
from cro_dt.VectorTree import dt_matrix_fit_dx
import cro_dt.VectorTree as vt
import cro_dt.cythonfns.TreeEvaluation as cy
import cro_dt.cythonfns.TreeEvalPython as cyp
import cro_dt.NumbaTree as nt
# import cro_dt.CupyTree as ct
import tensorflow as tf
import cro_dt.TensorflowTree as tft
import pstats, cProfile

if __name__ == "__main__":
    depth = 6

    n_samples = 10000
    n_classes = 10
    n_leaves = 2 ** depth
    n_features = 10

    simulations = 100

    X = np.random.uniform(-5, 5, (n_samples, n_features))
    X = np.int_(X).astype(np.double)
    y = np.random.randint(0, n_classes, n_samples)
    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.tile(y, (2 ** depth, 1))
    M = vt.create_mask_dx(depth)

    # tic = time.perf_counter()
    # for _ in range(simulations):
    #     W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
    #     W = vt.get_W_as_univariate(W)
    #
    #     acc_tree, _ = vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_)
    # toc = time.perf_counter()
    # print(f"Tree evaluation time: \t\t\t{(toc - tic)} s")

    W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))

    attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
    thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
    inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)

    W = vt.get_W_as_univariate(W)
    tic = time.perf_counter()
    for _ in range(simulations):
        acc_cytree, _ = cy.dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
    toc = time.perf_counter()
    print(f"Cytree evaluation time: \t\t{(toc - tic)} s")

    tic = time.perf_counter()
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)
        acc_matrix, _ = vt.dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
    toc = time.perf_counter()
    print(f"Matrix evaluation time: \t\t{(toc - tic)} s")

    total_time = 0
    for _ in range(simulations):
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)
        Xtf = tf.convert_to_tensor(X, dtype=tf.float64)
        Wtf = tf.convert_to_tensor(W, dtype=tf.float64)
        X_tf = tf.convert_to_tensor(X_, dtype=tf.float64)
        Y_tf = tf.convert_to_tensor(Y_, dtype=tf.int32)
        Mtf = tf.convert_to_tensor(M, dtype=tf.int32)

        tic = time.perf_counter()
        acc_tensorflow, _ = tft.dt_matrix_fit(X, None, Wtf, depth, n_classes, X_tf, Y_tf, Mtf, n_samples)
        toc = time.perf_counter()
        total_time += toc - tic
    print(f"Tensorflow evaluation time: \t{(total_time)} s")

    # def foobar():
    #     for _ in range(simulations):
    #         W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
    #         acc_cytree, _ = cy.dt_tree_fit(X, y, W, depth, n_classes, X_)
    #
    # import pyximport
    #
    # pyximport.install()
    #
    # cProfile.runctx("foobar()", globals(), locals(), "Profile.prof")
    #
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()