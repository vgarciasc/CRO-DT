import pdb
import numpy as np
import time
from cro_dt.VectorTree import dt_matrix_fit_dx
import cro_dt.VectorTree as vt
# import cro_dt.cythonfns.TreeEvaluation as cy
import cro_dt.NumbaTree as nt
import cro_dt.CupyTree as ct

if __name__ == "__main__":
    depth = 5

    n_samples = 10000
    n_classes = 2
    n_leaves = 2 ** depth
    n_features = 10

    # Testing whether the evaluation schemes are equal
    for _ in range(100):
        X = np.random.uniform(-5, 5, (n_samples, n_features))
        X = np.int_(X).astype(np.double)
        y = np.random.randint(0, n_classes, n_samples)
        W = np.random.uniform(-1, 1, (n_leaves - 1, n_features + 1))
        W = vt.get_W_as_univariate(W)

        X_ = np.vstack((np.ones(len(X)).T, X.T)).T
        Y_ = np.tile(y, (2 ** depth, 1))
        M = vt.create_mask_dx(depth)

        tic = time.perf_counter()
        for _ in range(10):
            acc_tree, _ = vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_)
        toc = time.perf_counter()
        print(f"Tree evaluation time: \t\t\t{(toc - tic)} s")

        # tic = time.perf_counter()
        # for _ in range(10):
        #     acc_cytree, _ = cy.dt_tree_fit(X, y, W, depth, n_classes, X_)
        # toc = time.perf_counter()
        # print(f"Cytree evaluation time: \t\t{(toc - tic)} s")

        tic = time.perf_counter()
        for _ in range(10):
            acc_matrix, _ = dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
        toc = time.perf_counter()
        print(f"Matrix evaluation time: \t\t{(toc - tic)} s")

        # tic = time.perf_counter()
        # for _ in range(10):
        #     acc_cupy, _ = ct.dt_matrix_fit(X, y, W, depth, n_classes, X_, Y_, M)
        # toc = time.perf_counter()
        # print(f"Cupy tree evaluation time: \t\t{(toc - tic)} s")

        nt.dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)
        tic = time.perf_counter()
        for _ in range(10):
            acc_numba, _ = nt.dt_matrix_fit_dx_numba(X, y, W, depth, n_classes, X_, Y_, M)
        toc = time.perf_counter()
        print(f"Numba matrix evaluation time: \t\t{(toc - tic)} s")

        print("-----------")
        print(f"ACCURACIES: (tree: {acc_tree},  matrix: {acc_matrix}, cupy: {acc_cupy}, numba: {acc_numba})")

        X = np.vstack((np.ones(len(X)), X.T)).T
        Z = np.sign(W @ X.T)

        # Check if they are all the same
        if acc_tree != acc_matrix or acc_tree != acc_numba or acc_tree != acc_cupy:
            print("Error!")
            pdb.set_trace()

    print("Evaluation schemes are equivalent.")