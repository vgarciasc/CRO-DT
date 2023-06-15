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
from cro_dt.experiments.es import run_es

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAO Evaluator')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-s', '--simulations', help="How many simulations to run?", required=True, type=int)
    parser.add_argument('-d', '--depth', help="What is the depth?", required=True, type=int)
    parser.add_argument('-g', '--n_gens', help="How many generations?", required=False, default=10e6, type=int)
    parser.add_argument('--lambda', help="Value of lambda", required=False, default=100, type=int)
    parser.add_argument('--mu', help="Value of mu", required=False, default=10, type=int)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save', help='Should write?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # create ID
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    command_line = str(args)
    command_line += "\n\npython -m cro_dt.experiments.es_simple " + \
                    " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    num_nodes = 2 ** (args['depth'] + 1) - 1

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
        X = X.astype(np.float64)
        y = y.astype(np.int64)

        log.append(())

        for simulation in range(args["simulations"]):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation,
                                                                stratify=y)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation,
                                                    stratify=y_test)

            # fit CMA
            max_rules = 2 ** (args['depth']) - 1
            tik = time.perf_counter()
            model = run_es(dataset, args['lambda'], args['mu'], args['n_gens'], args['depth'], simulation_id=simulation)
            time_elapsed = time.perf_counter() - tik

            # look at performance
            train_acc = model.evaluate(X_train, y_train)
            test_acc = model.evaluate(X_test, y_test)

            accuracies_in.append(train_acc)
            accuracies_out.append(test_acc)
            trees.append(model)
            tree_sizes.append(num_nodes)
            times_elapsed.append(time_elapsed)
            log[-1] = (dataset, accuracies_in, accuracies_out, trees, tree_sizes, times_elapsed)

            if args["verbose"]:
                print(
                    f"[magenta]Simulation [red]{simulation + 1}[/red]/[red]{args['simulations']}[/red] for dataset '{dataset}'.[/magenta]")
                print("      Training accuracy: {}".format(train_acc))
                print("      Testing accuracy: {}".format(test_acc))
                print("      # of nodes: {}".format(num_nodes))
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