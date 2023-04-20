import pdb
import numpy as np
import time
from cro_dt.VectorTree import dt_matrix_fit_dx
import cro_dt.VectorTree as vt
import cro_dt.MatrixTree as mt
import cro_dt.cythonfns.TreeEvaluation as cy
import cro_dt.NumbaTree as nt
# import cro_dt.CupyTree as ct
import tensorflow as tf
import cro_dt.TensorflowTree as tft

if __name__ == "__main__":
    depth = 6

    n_samples = 1000
    n_classes = 2
    n_leaves = 2 ** depth
    n_features = 5
    simulations = 100

    # Testing whether the evaluation schemes are equal
    for _ in range(100):
        X = np.random.uniform(-5, 5, (n_samples, n_features))
        X = np.int_(X).astype(np.double)
        y = np.random.randint(0, n_classes, n_samples)
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)

        X_ = np.vstack((np.ones(len(X)).T, X.T)).T
        Y_ = np.int_(np.tile(y, (2 ** depth, 1)))
        M = vt.create_mask_dx(depth)
        N = len(X_)

        Xtf = tf.convert_to_tensor(X, dtype=tf.float64)
        X_tf = tf.convert_to_tensor(X_, dtype=tf.float64)
        Y_tf = tf.convert_to_tensor(Y_, dtype=tf.int32)
        Wtf = tf.convert_to_tensor(W, dtype=tf.float64)
        Mtf = tf.convert_to_tensor(M, dtype=tf.int32)
        y_nb = y + 1
        Y_nb = np.int_(np.tile(y_nb, (2 ** depth, 1)))
        Y_nb = tf.convert_to_tensor(Y_nb, dtype=tf.int32)
        W_batch = np.array([W for _ in range(simulations)])
        W_batch_tf = tf.convert_to_tensor(W_batch, dtype=tf.float64)

        attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)

        M_t = vt.create_nodes_tree_mapper(depth)
        tic = time.perf_counter()
        for _ in range(simulations):
            acc_tree, _ = vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_, M_t)
        toc = time.perf_counter()
        print(f"Tree evaluation time: \t\t\t\t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_cytree, _ = cy.dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
        toc = time.perf_counter()
        print(f"Cytree evaluation time: \t\t\t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_matrix, _ = dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
        toc = time.perf_counter()
        print(f"Matrix evaluation time: \t\t\t\t{(toc - tic)} s")

        # tic = time.perf_counter()
        # for _ in range(simulations):
        #     acc_cupy, _ = ct.dt_matrix_fit(X, y, W, depth, n_classes, X_, Y_, M)
        # toc = time.perf_counter()
        # print(f"Cupy tree evaluation time: \t\t{(toc - tic)} s")

        # nt.dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)
        # tic = time.perf_counter()
        # for _ in range(simulations):
        #     acc_numba, _ = nt.dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)
        # toc = time.perf_counter()
        # print(f"Numba matrix evaluation time: \t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_tensorflow, _ = tft.dt_matrix_fit(Xtf, None, Wtf, depth, n_classes, X_tf, Y_tf, Mtf, N)
        toc = time.perf_counter()
        print(f"Tensorflow evaluation time: \t\t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_tensorflow_nb, _ = tft.dt_matrix_fit_nb(Xtf, None, Wtf, depth, n_classes, X_tf, Y_nb, Mtf, N, n_leaves)
        toc = time.perf_counter()
        print(f"Tensorflow NB evaluation time: \t\t\t{(toc - tic)} s")

        tic = time.perf_counter()
        accs_tensorflow_total, _ = tft.dt_matrix_fit_batch(Xtf, None, W_batch_tf, depth, n_classes, X_tf, Y_tf, Mtf, N)
        toc = time.perf_counter()
        acc_tensorflow_total = accs_tensorflow_total[0]
        print(f"Tensorflow batch evaluation time: \t\t{(toc - tic)} s")

        tic = time.perf_counter()
        accs_tensorflow_nb_total, _ = tft.dt_matrix_fit_batch_nb(Xtf, None, W_batch_tf, depth, n_classes,
                                                                 X_tf, Y_nb, Mtf, N, n_leaves, batch_size=simulations)
        toc = time.perf_counter()
        acc_tensorflow_nb_total = accs_tensorflow_nb_total[0]
        print(f"Tensorflow NB batch evaluation time: \t{(toc - tic)} s")

        print("-----------")
        print(f"ACCURACIES: (tree: {acc_tree}, cytree: {acc_cytree}, matrix: {acc_matrix}, "
              f"tensorflow: {acc_tensorflow}, tensorflow nb: {acc_tensorflow_nb}, "
              f" tensorflow total: {acc_tensorflow_total}, tensorflow nb total: {acc_tensorflow_nb_total})")

        # Check if they are all the same
        if not np.allclose(acc_tree, acc_cytree, acc_matrix, acc_tensorflow, acc_tensorflow_total):
        # if not np.allclose(acc_matrix, acc_tensorflow, acc_numba):
            print("Error!")
            pdb.set_trace()

    print("Evaluation schemes are equivalent.")