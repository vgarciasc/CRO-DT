import os
import pdb
import argparse
import time
from datetime import datetime

import cma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn import metrics
from scipy.io.arff import loadarff
import tensorflow as tf
import cro_dt.TensorflowTree as tft
import cro_dt.VectorTree as vt
from cro_dt.sup_configs import get_config

from rich.console import Console
from rich import print

from cro_dt.SoftTree import SoftTree

console = Console()

def save_log_to_file(filename, log, prefix=""):
    output_str = prefix
    for (dataset, accuracies_in, accuracies_out, trees, tree_sizes, times_elapsed) in log:
        output_str += f"Dataset '{dataset}':\n"
        output_str += f"  {len(accuracies_in)} simulations executed.\n"
        output_str += f"  Average training accuracy: {'{:.1f}'.format(np.mean(accuracies_in) * 100).replace('.', ',')} ± {'{:.1f}'.format(np.std(accuracies_in) * 100).replace('.', ',')}\n"
        output_str += f"  Average testing accuracy: {'{:.1f}'.format(np.mean(accuracies_out) * 100).replace('.', ',')} ± {'{:.1f}'.format(np.std(accuracies_out) * 100).replace('.', ',')}\n"
        output_str += f"  Average tree size: {'{:.1f}'.format(np.mean(tree_sizes)).replace('.', ',')} ± {'{:.1f}'.format(np.std(tree_sizes)).replace('.', ',')}\n"
        output_str += f"  Average time elapsed: {'{:.1f}'.format(np.mean(times_elapsed)).replace('.', ',')} ± {'{:.1f}'.format(np.std(times_elapsed)).replace('.', ',')}\n"
        output_str += f"  Max tree size: {np.max(tree_sizes)}\n"
        output_str += "-------\n\n"

    with open(f"results/{filename}", "w") as f:
        f.write(output_str)


def get_accuracy_tf(weights, X, y, depth, n_classes, X_, Y_, M, N, n_leaves):
    W = weights.reshape((n_leaves-1, -1))
    W = vt.get_W_as_univariate(W)
    Wt = tf.convert_to_tensor(W, dtype=tf.float64)

    acc, _ = tft.dt_matrix_fit_nb(X, y, Wt, depth, n_classes, X_, Y_, M, N, n_leaves)

    return 1 - float(acc)

def get_common_variables(config, X, y, depth):
    n_classes = config["n_classes"]
    n_attributes = config["n_attributes"]
    n_leaves = 2 ** depth
    M = vt.create_mask_dx(depth)
    N = X.shape[0]

    X_ = np.vstack((np.ones(len(X)).T, X.T)).T
    Y_ = np.int_(np.tile(y, (n_leaves, 1)))

    Xt = tf.convert_to_tensor(X, dtype=tf.float64)
    X_t = tf.convert_to_tensor(X_, dtype=tf.float64)
    Y_t = tf.convert_to_tensor(Y_ + 1, dtype=tf.int32)
    Mt = tf.convert_to_tensor(M, dtype=tf.int32)

    return Xt, X_t, Y_t, Mt, N, n_leaves, n_classes, n_attributes

def run_CMA(config, X, y, depth, n_evals=10e3):
    Xt, X_t, Y_t, Mt, N, n_leaves, n_classes, n_attributes = get_common_variables(config, X, y, depth)

    tree = SoftTree(num_attributes=n_attributes, num_classes=n_classes)
    tree.randomize(depth=depth)

    x0 = tree.weights.flatten()
    sigma0 = 1

    x, es = cma.fmin2(get_accuracy_tf, x0, sigma0,
                      options={'maxfevals': n_evals, 'bounds': [-1, 1], 'tolflatfitness': 1e3, 'seed': 0},
                      args=(Xt, y, depth, n_classes, X_t, Y_t, Mt, N, n_leaves))

    tree.weights = x.reshape((tree.num_nodes, tree.num_attributes + 1))
    return tree

def evaluate_once(config, tree, X, y, depth):
    W = vt.get_W_as_univariate(tree.weights)
    Xt, X_t, Y_t, Mt, N, n_leaves, n_classes, n_attributes = get_common_variables(config, X, y, depth)
    accuracy = get_accuracy_tf(W.flatten(), Xt, y, args['depth'], n_classes, X_t, Y_t, Mt, N, n_leaves)
    return 1 - accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAO Evaluator')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-s', '--simulations', help="How many simulations to run?", required=True, type=int)
    parser.add_argument('-d', '--depth', help="What is the depth?", required=True, type=int)
    parser.add_argument('-e', '--n_evals', help="How many evaluations?", required=False, default=10e6, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save', help='Should write?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # create ID
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    command_line = str(args)
    command_line += "\n\npython cma.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"

    datasets = [
        "breast_cancer",
        "car",
        "banknote",
        "balance",
        "acute-1",
        "acute-2",
        "transfusion",
        "climate",
        "sonar",
        "optical",
        "drybean",
        "avila",
        "wine-red",
        "wine-white",
    ]

    if args['dataset'].endswith("onwards"):
        dataset_start = datasets.index(args['dataset'][:-len("_onwards")])
        datasets = [d for d in datasets[dataset_start:]]
    elif args["dataset"] != "all":
        datasets = [args["dataset"]]

    log = []
    for dataset in datasets:
        config = get_config(dataset)
        console.rule(f"[red]Training dataset {dataset}[/red]")
        accuracies_in = []
        accuracies_out = []
        trees = []
        tree_sizes = []
        times_elapsed = []

        # read the dataset
        df = pd.read_csv(f"cro_dt/experiments/data/{dataset}.csv")
        feat_names = df.columns[:-1]
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

        log.append(())

        for simulation in range(args["simulations"]):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation,
                                                                stratify=y)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation,
                                                    stratify=y_test)

            # fit CMA
            max_rules = 2 ** (args['depth']) - 1
            tik = time.perf_counter()
            model = run_CMA(config, X_train, y_train, args['depth'], n_evals=args['n_evals'])
            time_elapsed = time.perf_counter() - tik

            # look at performance
            train_acc = evaluate_once(config, model, X_train, y_train, args['depth'])
            test_acc = evaluate_once(config, model, X_test, y_test, args['depth'])

            accuracies_in.append(train_acc)
            accuracies_out.append(test_acc)
            trees.append(model)
            tree_sizes.append(model.num_nodes)
            times_elapsed.append(time_elapsed)
            log[-1] = (dataset, accuracies_in, accuracies_out, trees, tree_sizes, times_elapsed)

            if args["verbose"]:
                print(
                    f"[magenta]Simulation [red]{simulation + 1}[/red]/[red]{args['simulations']}[/red] for dataset '{dataset}'.[/magenta]")
                print("      Training accuracy: {}".format(train_acc))
                print("      Testing accuracy: {}".format(test_acc))
                print("      # of nodes: {}".format(model.num_nodes))
                # print(model.tree)

            if args["should_save"]:
                save_log_to_file(f"log_cma_d{args['depth']}_{curr_time}.txt", log, command_line)

        print("--------------------------")
        print(f"[yellow]Dataset {dataset}:[/yellow]")
        print(
            f"  Average accuracy in-sample: {'{:.3f}'.format(np.mean(accuracies_in))} ± {'{:.3f}'.format(np.std(accuracies_in))}")
        print(
            f"  Average accuracy out-of-sample: {'{:.3f}'.format(np.mean(accuracies_out))} ± {'{:.3f}'.format(np.std(accuracies_out))}")
        print(f"  Average tree size: {np.mean(tree_sizes)}")
        print(f"  Max tree size: {np.max(tree_sizes)}")