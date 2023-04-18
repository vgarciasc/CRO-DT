import pdb
from datetime import datetime

import numpy as np
import json
import time
import cro_dt.VectorTree as vt
import cro_dt.MatrixTree as mt
import cro_dt.cythonfns.TreeEvaluation as cy
import cro_dt.NumbaTree as nt
import tensorflow as tf
import cro_dt.TensorflowTree as tft
from cro_dt.plots.plot_time_perf import plot_time_perf

from rich import print

def generate_W(n_leaves, n_attributes):
    W = np.random.uniform(-1, 1, (n_leaves - 1, n_attributes + 1))
    W = vt.get_W_as_univariate(W)
    return W

def evaluate_algo_once(algo, X, y, X_, Y_, M, depth, n_attributes, n_classes, n_evals,
                       simul_idx=-1, n_simulations=-1, verbose=True):
    total_time = 0

    Xt = tf.convert_to_tensor(X, dtype=tf.float64)
    X_t = tf.convert_to_tensor(X_, dtype=tf.float64)
    Y_t = tf.convert_to_tensor(Y_, dtype=tf.int32)
    Mt = tf.convert_to_tensor(M, dtype=tf.int32)

    if verbose:
        print(f"------------------")
        print(f"Evaluating '{algo}' / {n_evals} (Depth {depth}, Simulation {simul_idx} / {n_simulations})")

    for eval in range(n_evals):
        if verbose and eval % (n_evals // 10) == 0:
            print(f"\t\t{eval} / {n_evals}")

        W = generate_W(n_leaves, n_attributes)
        Wt = tf.convert_to_tensor(W, dtype=tf.float64)
        attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
        inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)

        tic = time.perf_counter()
        if algo == "tree":
            vt.dt_tree_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
        elif algo == "cytree":
            cy.dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
        elif algo == "matrix":
            vt.dt_matrix_fit_dx(X, y, W, depth, n_classes, X_, Y_, M)
        elif algo == "tf":
            tft.dt_matrix_fit(Xt, None, Wt, depth, n_classes, X_t, Y_t, Mt)

        toc = time.perf_counter()
        total_time += (toc - tic)

    if algo == "tf_batch":
        W_batch = np.array([generate_W(n_leaves, n_attributes) for _ in range(n_evals)])
        W_batch = tf.convert_to_tensor(W_batch, dtype=tf.float64)

        # Warming up the GPU, first time is always slow
        tft.dt_matrix_fit_batch(Xt, None, W_batch, depth, n_classes, X_t, Y_t, Mt)

        tic = time.perf_counter()
        tft.dt_matrix_fit_batch(Xt, None, W_batch, depth, n_classes, X_t, Y_t, Mt)
        toc = time.perf_counter()
        total_time += (toc - tic)

    if verbose:
        print(f"\tTotal time: {total_time:.2f} seconds")

    return total_time

def evaluate_algo(algo, X, y, X_, Y_, M, depth, n_attributes, n_classes, n_evals, n_simulations, verbose=True):
    return [evaluate_algo_once(algo, X, y, X_, Y_, M,
                               depth, n_attributes,
                               n_classes, n_evals,
                               simul_idx, n_simulations, verbose) for simul_idx in range(n_simulations)]

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    n_evals = int(1000)
    n_simulations = 10
    datasets = [(100, 3, 2), (1000, 3, 2), (1000, 3, 10), (1000, 10, 10),
                (10000, 3, 2), (10000, 3, 10), (100000, 10, 10)]

    time_measures = {}

    for N, n_classes, n_attributes in datasets:
        dataset_str = f"{N}_{n_classes}_{n_attributes}"
        time_measures[dataset_str] = {}

        X = np.random.uniform(-5, 5, (N, n_attributes))
        X_ = np.vstack((np.ones(len(X)).T, X.T)).T
        y = np.random.randint(0, n_classes, N)

        for depth in range(2, 8):
            n_leaves = 2 ** depth
            Y_ = np.int_(np.tile(y, (n_leaves, 1)))
            M = vt.create_mask_dx(depth)

            time_measures[dataset_str][f"depth_{depth}"] = {}

            for algo in ["cytree", "matrix", "tf", "tf_batch"]:
                measured_time = evaluate_algo(algo, X, y, X_, Y_, M, depth, n_attributes, n_classes, n_evals, n_simulations)
                time_measures[dataset_str][f"depth_{depth}"][algo] = measured_time
                with open(f"results/time_measures_{timestamp}.json", "w") as f:
                    json.dump(time_measures, f, indent=2)

        print(time_measures)

    # Plot data
    plot_time_perf(time_measures)