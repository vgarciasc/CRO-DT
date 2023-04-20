import pdb
from datetime import datetime

import argparse
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


def evaluate_algo_once(algo, X, y, X_, Y_, M, depth, n_attributes, n_classes, n_gens, gen_size,
                       simul_idx=-1, n_simulations=-1, dataset_str='undefined', verbose=True):
    Xt = tf.convert_to_tensor(X, dtype=tf.float64)
    X_t = tf.convert_to_tensor(X_, dtype=tf.float64)
    Y_t = tf.convert_to_tensor(Y_, dtype=tf.int32)
    Mt = tf.convert_to_tensor(M, dtype=tf.int32)

    total_time = 0

    if verbose:
        print(f"------------------")
        print(f"Evaluating '{algo}' (Dataset: {dataset_str}, Depth {depth}, Simulation {simul_idx} / {n_simulations})")

    for gen in range(n_gens):
        if verbose and gen % (n_gens // 10) == 0:
            print(f"Gen: \t\t{gen} / {n_gens}")

        if algo in ["tf_batch", "tf_batch_cpu"]:
            W_batch = np.array([generate_W(n_leaves, n_attributes) for _ in range(gen_size)])
            W_batch = tf.convert_to_tensor(W_batch, dtype=tf.float64)

            with tf.device("/CPU:0" if algo == "tf_batch_cpu" else "/GPU:0"):
                # Warming up the GPU, first time is always slow
                tft.dt_matrix_fit_batch(Xt, None, W_batch, depth, n_classes, X_t, Y_t, Mt)

                tic = time.perf_counter()
                tft.dt_matrix_fit_batch(Xt, None, W_batch, depth, n_classes, X_t, Y_t, Mt)
                toc = time.perf_counter()
                total_time += (toc - tic)
        else:
            for _ in range(gen_size):
                W = generate_W(n_leaves, n_attributes)
                Wt = tf.convert_to_tensor(W, dtype=tf.float64)
                attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
                thresholds = np.array(
                    [(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if
                     val != 0 and i != 0])
                inversions = np.array(
                    [(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0],
                    dtype=np.int64)
                M_t = vt.create_nodes_tree_mapper(depth)

                if algo == "tf":
                    # Warming up the GPU, first time is always slow
                    tft.dt_matrix_fit(Xt, None, Wt, depth, n_classes, X_t, Y_t, Mt)

                tic = time.perf_counter()
                if algo == "tree":
                    vt.dt_tree_fit_paper(X, y, W, depth, n_classes, X_, Y_, M_t)
                elif algo == "cytree":
                    cy.dt_tree_fit(X_, y, W, depth, n_classes, attributes, thresholds, inversions)
                elif algo == "matrix":
                    vt.dt_matrix_fit_paper(X, y, W, depth, n_classes, X_, Y_, M)
                elif algo == "tf":
                    tft.dt_matrix_fit(Xt, None, Wt, depth, n_classes, X_t, Y_t, Mt)
                elif algo == "tf_cpu":
                    with tf.device("/CPU:0"):
                        tft.dt_matrix_fit(Xt, None, Wt, depth, n_classes, X_t, Y_t, Mt)

                toc = time.perf_counter()
                total_time += (toc - tic)

    if verbose:
        print(f"\tTotal time: {total_time:.2f} seconds")

    return total_time


def evaluate_algo(algo, X, y, X_, Y_, M, depth, n_attributes, n_classes, n_gens, gen_size, n_simulations, verbose=True):
    print("-----------------------------")
    return [evaluate_algo_once(algo, X, y, X_, Y_, M,
                               depth, n_attributes, n_classes,
                               n_gens, gen_size, simul_idx, n_simulations,
                               dataset_str, verbose) for simul_idx in range(n_simulations)]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-n", "--n_gens", type=int, default=10)
    argparser.add_argument("-g", "--gen_size", type=int, default=10)
    argparser.add_argument("-s", "--n_simulations", type=int, default=100)
    argparser.add_argument("-o", "--output", type=str, default="undefined")
    argparser.add_argument("-a", "--algorithms", type=str, default="['cytree', 'matrix', 'tf', 'tf_batch']")
    args = argparser.parse_args()

    if args.output == "undefined":
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        filename = f"results/time_measures_{timestamp}.json"
    else:
        filename = args.output

    n_gens = args.n_gens
    gen_size = args.gen_size
    n_simulations = args.n_simulations
    algorithms = eval(args.algorithms)
    datasets = [(100, 3, 2), (1000, 3, 2), (1000, 3, 10), (1000, 10, 10),
                (10000, 3, 2), (10000, 3, 10), (100000, 10, 10)]

    time_measures = {}

    for algo in algorithms:
        time_measures[algo] = {}

        for N, n_classes, n_attributes in datasets:
            dataset_str = f"{N}_{n_classes}_{n_attributes}"

            X = np.random.uniform(-5, 5, (N, n_attributes))
            X_ = np.vstack((np.ones(len(X)).T, X.T)).T
            # We assume that the labels start at 1
            y = np.random.randint(0, n_classes, N)

            time_measures[algo][dataset_str] = {}

            for depth in range(2, 8):
                n_leaves = 2 ** depth
                Y_ = np.int_(np.tile(y, (n_leaves, 1)))
                M = vt.create_mask_dx(depth)

                try:
                    measured_time = evaluate_algo(algo, X, y, X_, Y_, M, depth,
                                                  n_attributes, n_classes, n_gens, gen_size,
                                                  n_simulations, dataset_str)
                except Exception as e:
                    print(f"Error: {e}")
                    measured_time = [0]

                time_measures[algo][dataset_str][f"depth_{depth}"] = measured_time

                with open(filename, "w") as f:
                    json.dump(time_measures, f, indent=2)
                print(f"Saved to '{filename}'")

        print(time_measures)