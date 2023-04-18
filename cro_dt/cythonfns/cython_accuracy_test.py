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
    depth = 2

    n_samples = 100
    n_classes = 3
    n_leaves = 2 ** depth
    n_features = 2
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

        attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)

        # tic = time.perf_counter()
        # for _ in range(simulations):
        #     acc_tree, _ = vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_)
        # toc = time.perf_counter()
        # print(f"Tree evaluation time: \t\t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_cytree, _ = cy.dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
        toc = time.perf_counter()
        print(f"Cytree evaluation time: \t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(simulations):
            acc_matrix, _ = dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
        toc = time.perf_counter()
        print(f"Matrix evaluation time: \t\t{(toc - tic)} s")

        tic = time.perf_counter()
        W_total = np.array([W for _ in range(simulations)])
        accs_matrix_2, _ = mt.dt_matrix_fit_total(X_, Y_, W_total, M, depth, n_classes)
        acc_matrix_2 = accs_matrix_2[0]
        toc = time.perf_counter()
        print(f"Matrix total evaluation time: \t\t{(toc - tic)} s")

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
        with tf.device("/GPU:0"):
            X = tf.convert_to_tensor(X, dtype=tf.float64)
            W = tf.convert_to_tensor(W, dtype=tf.float64)
            X_ = tf.convert_to_tensor(X_, dtype=tf.float64)
            Y_ = tf.convert_to_tensor(Y_, dtype=tf.int32)
            M = tf.convert_to_tensor(M, dtype=tf.int32)

            for _ in range(simulations):
                acc_tensorflow, _ = tft.dt_matrix_fit(X, None, W, depth, n_classes, X_, Y_, M)
        toc = time.perf_counter()
        print(f"Tensorflow evaluation time: \t{(toc - tic)} s")

        tic = time.perf_counter()
        W_total = np.array([W for _ in range(simulations)])
        accs_tensorflow_total, _ = tft.dt_matrix_fit_batch(X, None, W_total, depth, n_classes, X_, Y_, M)
        acc_tensorflow_total = accs_tensorflow_total[0]
        toc = time.perf_counter()
        print(f"Tensorflow total evaluation time: \t{(toc - tic)} s")

        print("-----------")
        print(f"ACCURACIES: (cytree: {acc_cytree},  matrix: {acc_matrix}, matrix2: {acc_matrix_2}, tensorflow: {acc_tensorflow}, tensorflow total: {acc_tensorflow_total})")
        # print(f"ACCURACIES: (matrix: {acc_matrix}, tensorflow: {acc_tensorflow}, numba = {acc_numba}")

        # Check if they are all the same
        if not np.allclose(acc_cytree, acc_matrix, acc_matrix_2, acc_tensorflow, acc_tensorflow_total):
        # if not np.allclose(acc_matrix, acc_tensorflow, acc_numba):
            print("Error!")
            pdb.set_trace()

    print("Evaluation schemes are equivalent.")