import os
import pdb
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn import metrics
from scipy.io.arff import loadarff

# installable with: `pip install imodels`
from imodels import C45TreeClassifier, GreedyTreeClassifier, TaoTreeClassifier
from imodels.util.convert import tree_to_code

from rich.console import Console
from rich import print
console = Console()

def save_log_to_file(filename, log, prefix=""):
    output_str = prefix
    for (dataset, accuracies_in, accuracies_out, trees, tree_sizes) in log:
        output_str += f"Dataset '{dataset}':\n"
        output_str += f"  {len(accuracies_in)} simulations executed.\n"
        output_str += f"  Average training accuracy: {'{:.1f}'.format(np.mean(accuracies_in)* 100).replace('.',',')} ± {'{:.1f}'.format(np.std(accuracies_in)* 100).replace('.',',')}\n"
        output_str += f"  Average testing accuracy: {'{:.1f}'.format(np.mean(accuracies_out)* 100).replace('.',',')} ± {'{:.1f}'.format(np.std(accuracies_out)* 100).replace('.',',')}\n"
        output_str += f"  Average tree size: {'{:.1f}'.format(np.mean(tree_sizes)).replace('.',',')} ± {'{:.1f}'.format(np.std(tree_sizes)).replace('.',',')}\n"
        output_str += f"  Max tree size: {np.max(tree_sizes)}\n"
        output_str += "-------\n\n"

    with open(f"results/{filename}", "w") as f:
        f.write(output_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAO Evaluator')
    parser.add_argument('-i','--dataset',help="What dataset to use?", required=True, type=str)
    parser.add_argument('-s','--simulations',help="How many simulations to run?", required=True, type=int)
    parser.add_argument('-d','--depth',help="What is the depth?", required=True, type=int)
    parser.add_argument('-r','--regularization',help="What is the regularization?", required=False, default=0.0001, type=float)
    # parser.add_argument('-t','--test_percentage',help="What percentage of dataset for testing?", required=False, default=0.0001, type=float)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_save', help='Should write?', required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # create ID
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    command_line = str(args)
    command_line += "\n\npython c45.py " + " ".join([f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

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
        console.rule(f"[red]Training dataset {dataset}[/red]")
        accuracies_in = []
        accuracies_out = []
        trees = []
        tree_sizes = []

        # read the dataset
        df = pd.read_csv(f"data/{dataset}.csv")
        feat_names = df.columns[:-1]
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

        log.append(())

        for simulation in range(args["simulations"]):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation, stratify=y)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation, stratify=y_test)
            
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=simulation, stratify=y)

            # scaler = StandardScaler().fit(X_train)
            # X_train = scaler.transform(X_train)
            # X_test = scaler.transform(X_test)

            # fit C45
            max_rules = 2 ** (args['depth']) - 1
            model = C45TreeClassifier(max_rules=max_rules)
            model.fit(X_train, y_train, feature_names=feat_names)
            # print(model)

            # look at performance
            probs = model.predict_proba(X_test)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_acc = np.mean([1 if y_pred_train[i] == y_train[i] else 0 for i in range(len(y_train))])
            test_acc = np.mean([1 if y_pred_test[i] == y_test[i] else 0 for i in range(len(y_test))])
            n_nodes = model.complexity_

            accuracies_in.append(train_acc)
            accuracies_out.append(test_acc)
            trees.append(model)
            tree_sizes.append(n_nodes)
            log[-1] = (dataset, accuracies_in, accuracies_out, trees, tree_sizes)
            
            if args["verbose"]:
                print(f"[magenta]Simulation [red]{simulation + 1}[/red]/[red]{args['simulations']}[/red] for dataset '{dataset}'.[/magenta]")
                print("      Training accuracy: {}".format(train_acc))
                print("      Testing accuracy: {}".format(test_acc))
                print("      # of nodes: {}".format(n_nodes))
                # print(model.tree)

            if args["should_save"]:
                save_log_to_file(f"log_c45_{curr_time}.txt", log, command_line)

        print("--------------------------")
        print(f"[yellow]Dataset {dataset}:[/yellow]")
        print(f"  Average accuracy in-sample: {'{:.3f}'.format(np.mean(accuracies_in))} ± {'{:.3f}'.format(np.std(accuracies_in))}")
        print(f"  Average accuracy out-of-sample: {'{:.3f}'.format(np.mean(accuracies_out))} ± {'{:.3f}'.format(np.std(accuracies_out))}")
        print(f"  Average tree size: {np.mean(tree_sizes)}")
        print(f"  Max tree size: {np.max(tree_sizes)}")