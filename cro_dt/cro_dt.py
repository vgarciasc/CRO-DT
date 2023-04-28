import pdb
import math
import argparse
import copy
import sys
from datetime import datetime
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from AbsObjetiveFunc import AbsObjetiveFunc
from CRO_SL import CRO_SL
from CoralPopulation import Coral
from SubstrateReal import SubstrateReal
import time
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from rich import print

from cro_dt.utils import printv
import cro_dt.VectorTree as vt
# import cro_dt.CupyTree as cp
import cro_dt.TensorflowTree as tft
import cro_dt.cythonfns.TreeEvaluation as cy
from cro_dt.sup_configs import get_config, load_dataset, artificial_dataset_list, real_dataset_list
from cro_dt.cart import get_cart_as_W

def get_initial_pop(data_config, popsize, X_train, y_train,
                    should_cart_init, desired_depth, cart_pop_filepath, objfunc):
    # Creating CART population
    cart_pop = []
    if cart_pop_filepath != None:
        with open(cart_pop_filepath, "rb") as f:
            cart_pops = pickle.load(f)

        for (dataset_code, pop) in cart_pops:
            if dataset_code == data_config["code"]:
                cart_pop = pop
    elif should_cart_init:
        for alpha in np.linspace(0, 0.5, 1000):
            for criterion in ["gini", "entropy", "log_loss"]:
                dt = DecisionTreeClassifier(ccp_alpha=alpha, criterion=criterion)
                dt.fit(X_train, y_train)

                if dt.get_depth() == desired_depth:
                    W = get_cart_as_W(data_config, dt, desired_depth)
                    cart_pop.append(W)

        cart_pop = np.unique(cart_pop, axis=0)
        cart_pop = [f for f in cart_pop]
        print(f"Different CART solutions found: {len(cart_pop)}")

        for _ in range(len(cart_pop), popsize // 3):
            dt = DecisionTreeClassifier(max_depth=desired_depth, splitter="random")
            dt.fit(X_train, y_train)

            W = get_cart_as_W(data_config, dt, desired_depth)
            cart_pop.append(W)

        if len(cart_pop) > popsize // 3:
            cart_pop = cart_pop[:popsize // 3]

    # Creating mutated CART population
    mutated_cart_pop = []
    for cart in cart_pop:
        mutated_cart = cart + np.random.normal(0, 1, size=cart.shape)
        mutated_cart_pop.append(mutated_cart)

    # Creating random population based on vectors
    random_pop_continuous = []
    for _ in range(len(cart_pop) + len(mutated_cart_pop), popsize):
        random_pop_continuous.append(objfunc.random_solution())

    return cart_pop + mutated_cart_pop + random_pop_continuous


def get_W_from_solution(solution, depth, n_attributes, args):
    # if args["evaluation_scheme"].startswith("tf") and args["evaluation_scheme"] != "tf_batch":
    #     W = tf.cast(tf.reshape(solution, [2 ** depth - 1, n_attributes + 1]), dtype=tf.float64)
    # else:
    W = solution.reshape((2 ** depth - 1, n_attributes + 1))

    if args["univariate"]:
        W = vt.get_W_as_univariate(W)

    if args["should_use_threshold"]:
        W[:, 1:][abs(W[:, 1:]) < args["threshold"]] = 0
        W[:, 0][W[:, 0] == 0] += 0.01

    return W

def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix + "\n"
    string_full = prefix + "\n"

    for config, history in zip(configs, histories):
        elapsed_times, scalers, multiv_info, univ_info = zip(*history)
        multiv_acc_in, multiv_acc_test, multiv_W, multiv_labels = zip(*multiv_info)
        univ_acc_in, univ_acc_test, univ_W, univ_labels = zip(*univ_info)

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"{len(elapsed_times)} simulations executed.\n"
        string_summ += f"Average in-sample multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_in))} ± {'{:.3f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average in-sample univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_in))} ± {'{:.3f}'.format(np.std(univ_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_test))} ± {'{:.3f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += f"Average test univariate accuracy: {'{:.3f}'.format(np.mean(univ_acc_test))} ± {'{:.3f}'.format(np.std(univ_acc_test))}\n"
        string_summ += "\n"
        string_summ += f"Best test multivariate accuracy: {'{:.3f}'.format(multiv_acc_test[np.argmax(multiv_acc_test)])}\n"
        string_summ += f"Best test univariate accuracy: {'{:.3f}'.format(univ_acc_test[np.argmax(univ_acc_test)])}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += "--------------------------------------------------\n\n"
        string_full += f"DATASET: {config['name']}\n"

        for (elapsed_time, \
             (scaler), \
             (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
             (univ_acc_in, univ_acc_test, univ_labels, univ_W)) in history:
            string_full += f"In-sample:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_in}" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_in}" + "\n"
            string_full += f"Test:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_test}" + "\n"
            string_full += f"        Univariate accuracy: {univ_acc_test}" + "\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            string_full += "\n"
            string_full += "Multivariate tree:\n" + vt.weights2treestr(multiv_W, multiv_labels, config,
                                                                       use_attribute_names=False, scaler=scaler)
            string_full += f"\n"
            string_full += f"Multivariate labels: {multiv_labels}\n"
            string_full += str(multiv_W)
            string_full += f"\n\n"
            string_full += "Univariate tree:\n" + vt.weights2treestr(univ_W, univ_labels, config,
                                                                     use_attribute_names=False, scaler=scaler)
            string_full += f"\n"
            string_full += f"Univariate labels: {univ_labels}\n"
            string_full += str(univ_W)
            string_full += "\n\n--------\n\n"

    with open(output_path_summary, "w", encoding="utf-8") as text_file:
        text_file.write(string_summ)

    with open(output_path_full, "w", encoding="utf-8") as text_file:
        text_file.write(string_full)


def get_substrates_real(cro_configs):
    substrates = []
    for substrate_real in cro_configs["substrates_real"]:
        substrates.append(SubstrateReal(substrate_real["name"], substrate_real["params"]))
    return substrates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRO-SL for Supervised Tree Induction')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-c', '--cro_config', help="How many function evaluations to stop at?", required=True, type=str)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=True, type=int)
    parser.add_argument('-d', '--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('--initial_pop', help="File with initial population", required=False, default=None, type=str)
    parser.add_argument('--univariate', help='Should use univariate tree\'s accuracy when measuring fitness?',
                        required=False, default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--alpha', help="How to penalize tree multivariateness?", required=False, default=1.0,
                        type=float)
    parser.add_argument('--should_normalize_rows', help='Should normalize rows?', required=False, default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_cart_init', help='Should initialize with CART trees?', required=False, default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_normalize_penalty', help='Should normalize penalty?', required=False, default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_get_best_from_validation', help='Should get best solution from validation set?',
                        required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_apply_exponential', help='Should apply exponential penalty?', required=False,
                        default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--should_use_threshold', help='Should ignore weights under a certain threshold?',
                        required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--threshold', help="Under which threshold should weights be ignored?", required=False,
                        default=0.05, type=float)
    parser.add_argument('--should_save_reports', help='Should save PCRO-SL reports?', required=False, default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--start_from', help='Should start from where?', required=False, default=0, type=int)
    parser.add_argument('--evaluation_scheme', help='Which evaluation scheme to use?', required=False, default="dx",
                        type=str)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="log", type=str)
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # Initialization
    depth = args["depth"]
    alpha = args["alpha"]
    mask = vt.create_mask(depth)
    with open(args["cro_config"]) as f:
        cro_configs = json.load(f)
    popsize = cro_configs["general"]["popSize"]
    n_inners, n_leaves = 2 ** depth - 1, 2 ** depth

    command_line = str(args)
    command_line += "\n\npython -m cro_dt.cro_dt " + " ".join(
        [f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    command_line += str(cro_configs)
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path_summ = f"results/{args['output_prefix']}_{curr_time}_summary.txt"
    output_path_full = f"results/{args['output_prefix']}_{curr_time}_full.txt"

    if args['dataset'].startswith("artificial"):
        dataset_list = artificial_dataset_list
    else:
        dataset_list = real_dataset_list

    if args['dataset'].endswith('all'):
        data_configs = [get_config(d) for d in dataset_list]
    elif args['dataset'].endswith("onwards"):
        dataset_start = dataset_list.index(args['dataset'][:-len("_onwards")])
        data_configs = [get_config(d) for d in dataset_list[dataset_start:]]
    else:
        data_configs = [get_config(args['dataset'])]

    if args["evaluation_scheme"] == "tree":
        M = vt.create_nodes_tree_mapper(depth)
    else:
        M = vt.create_mask_dx(depth)

        if args["evaluation_scheme"] in ["tensorflow", "tensorflow_cpu", "tensorflow_total"]:
            M = tf.convert_to_tensor(M, dtype=tf.int32)

    histories = []
    for dataset_id, data_config in enumerate(data_configs):
        X, y = load_dataset(data_config)
        N = len(X)

        n_attributes = data_config["n_attributes"]
        n_classes = data_config["n_classes"]
        max_penalty = (n_attributes - 1) * (2 ** depth - 1)

        histories.append([])

        start_idx = args['start_from'] if data_config == data_configs[0] else 0
        simulations = range(args["simulations"])[start_idx:]

        for simulation in simulations:
            if args["dataset"].startswith("artificial"):
                X_train, y_train = X, y
                X_test, y_test = X, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation, stratify=y)
                X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation, stratify=y_test)

            if args["should_normalize_dataset"]:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                scaler = None

            # Preparing matrices that need to be calculated only once
            X_train_ = np.vstack((np.ones(len(X_train)).T, X_train.T)).T
            Y_train_ = np.tile(y_train, (2 ** depth, 1))
            Xt = tf.convert_to_tensor(X_train, dtype=tf.float64)
            X_t = tf.convert_to_tensor(X_train_, dtype=tf.float64)
            Y_t = tf.convert_to_tensor(Y_train_ + 1, dtype=tf.int32)
            Mt = tf.convert_to_tensor(M, dtype=tf.int32)

            print("=" * 50)
            print(
                f"[red]Iteration {simulation}/{args['simulations']}[/red] [yellow](dataset: {data_config['name']}, {dataset_id}/{len(data_configs)}):[/yellow]")
            print("=" * 50)

            # Creating objective function based on execution parameters
            class SupervisedObjectiveFunc(AbsObjetiveFunc):
                def __init__(self, size, opt="max"):
                    self.size = size
                    self.algo = args["evaluation_scheme"]
                    super().__init__(self.size, opt)

                def objetive(self, solution):
                    W = get_W_from_solution(solution, depth, n_attributes, args)

                    if self.algo == "tree":
                        M_tree = vt.create_nodes_tree_mapper(depth)
                        accuracy, _ = vt.dt_tree_fit_paper(X_train, y_train, W, depth, n_classes, X_train_, Y_train_, M_tree)

                    elif self.algo == "cytree":
                        attributes = np.array([i for w in W for i, val in enumerate(w) if val != 0 and i != 0])
                        thresholds = np.array([(w[0] / val if val < 0 else - w[0] / val) for w in W for i, val in enumerate(w) if val != 0 and i != 0])
                        inversions = np.array([(-1 if val < 0 else 1) for w in W for i, val in enumerate(w) if val != 0 and i != 0], dtype=np.int64)
                        accuracy, _ = cy.dt_tree_fit(X_train_, y_train, W, depth, n_classes, attributes, thresholds, inversions)

                    elif self.algo == "matrix":
                        accuracy, _ = vt.dt_matrix_fit_paper(X_train, y_train, W, depth, n_classes, X_train_, Y_train_, M)

                    elif self.algo == "tf":
                        accuracy, _ = tft.dt_matrix_fit_nb(Xt, None, W, depth, n_classes, X_t, Y_t, Mt, N, n_leaves)

                    elif self.algo == "tf_cpu":
                        with tf.device("/CPU:0"):
                            accuracy, _ = tft.dt_matrix_fit_nb(Xt, None, W, depth, n_classes, X_t, Y_t, Mt, N, n_leaves)

                    if args["univariate"]:
                        return accuracy
                    else:
                        penalty = vt.get_penalty(W, max_penalty, alpha=args["alpha"],
                                                 should_normalize_rows=args["should_normalize_rows"], \
                                                 should_normalize_penalty=args["should_normalize_penalty"], \
                                                 should_apply_exp=args["should_apply_exponential"])
                        return accuracy - penalty

                def random_solution(self):
                    # if args["evaluation_scheme"].startswith("tf"):
                    #     return tf.random.uniform(shape=[(2**depth - 1) * (n_attributes + 1)], minval=-1, maxval=1)
                    # else:
                    return vt.generate_random_weights(n_attributes, depth)

                def check_bounds(self, solution):
                    # if args["evaluation_scheme"].startswith("tf") and args["evaluation_scheme"] != "tf_batch":
                    #     return tf.clip_by_value(solution, -1, 1)
                    # else:
                    return np.clip(solution.copy(), -1, 1)

            sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
            objfunc = SupervisedObjectiveFunc(sol_size)
            c = CRO_SL(objfunc, get_substrates_real(cro_configs), cro_configs["general"])

            # Getting initial population
            if not args["dataset"].startswith("artificial"):
                args['initial_pop'] = None if args['initial_pop'] == 'None' else args['initial_pop']
                initial_pop = get_initial_pop(data_config, popsize, X_train, y_train,
                                              args["should_cart_init"], args["depth"], args["initial_pop"], objfunc)

                if initial_pop is not None:
                    c.population.population = []
                    for tree in initial_pop:
                        coral = Coral(tree.flatten(), objfunc=objfunc)
                        coral.get_fitness()
                        c.population.population.append(coral)

                print(f"Average accuracy in CART seeding: {np.mean([f.fitness for f in c.population.population])}")
                print(f"Best accuracy in CART seeding: {np.max([f.fitness for f in c.population.population])}")

            # Running CRO-DT
            start_time = time.perf_counter()
            c.data = (X_t, Y_t, Mt, depth, n_attributes, n_classes)
            c.evaluation_scheme = args["evaluation_scheme"]
            c.is_univariate = args["univariate"]

            _, fit = c.optimize()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Save reports to disk
            if args['should_save_reports']:
                report_filename = f"results/{args['output_prefix']}_{data_config['code']}_{simulation}.png"
                c.display_report(show_plots=False, filename=report_filename)
                print(f"Saved report figure to '{report_filename}'.")

            # Post-process returned model from CRO-DT
            multiv_W, _ = c.population.best_solution()
            multiv_W = multiv_W.reshape((2 ** depth - 1, n_attributes + 1))

            if args["should_use_threshold"]:
                multiv_W[:, 1:][abs(multiv_W[:, 1:]) < args["threshold"]] = 0
                multiv_W[:, 0][multiv_W[:, 0] == 0] += 0.01

            univ_W = vt.get_W_as_univariate(multiv_W)
            if args["univariate"]:
                multiv_W = vt.get_W_as_univariate(univ_W)

            _, multiv_labels = vt.dt_matrix_fit_dx(X_train, y_train, multiv_W, depth, n_classes)
            multiv_acc_in = vt.calc_accuracy(X_train, y_train, multiv_W, multiv_labels)
            multiv_acc_test = vt.calc_accuracy(X_test, y_test, multiv_W, multiv_labels)

            _, univ_labels = vt.dt_matrix_fit_dx(X_train, y_train, univ_W, depth, n_classes)
            univ_acc_in = vt.calc_accuracy(X_train, y_train, univ_W, univ_labels)
            univ_acc_test = vt.calc_accuracy(X_test, y_test, univ_W, univ_labels)

            histories[-1].append((elapsed_time,
                                  (scaler), \
                                  (multiv_acc_in, multiv_acc_test, multiv_labels, multiv_W), \
                                  (univ_acc_in, univ_acc_test, univ_labels, univ_W)))
            save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)

        print(f"Saved summary to '{output_path_summ}'.")
        print(f"Saved full data to '{output_path_full}'.")
