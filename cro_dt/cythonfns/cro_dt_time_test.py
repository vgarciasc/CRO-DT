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

from cro_dt.cro_dt import get_W_from_solution, get_substrates_real

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--evaluation_scheme', help='Which evaluation scheme to use?', required=False,
                        default="dx", type=str)
    args = vars(parser.parse_args())

    if args["evaluation_scheme"] == "matrix":
        fitness_evaluation = vt.dt_matrix_fit_dx
    elif args["evaluation_scheme"] == "tree":
        fitness_evaluation = vt.dt_tree_fit_dx
    elif args["evaluation_scheme"] == "cytree":
        fitness_evaluation = cy.dt_tree_fit
    elif args["evaluation_scheme"] == "tensorflow":
        fitness_evaluation = tft.dt_matrix_fit
    else:
        print(f"Value '{args['evaluation_scheme']}' for 'evaluation scheme' is invalid.")
        sys.exit(0)

    if tf.config.list_physical_devices('GPU'):
        print("[green]USING GPU[/green]")
    else:
        print("[red]NOT USING GPU[/red]")

    depth = 4
    data_config = get_config("artificial_1000_3_2")
    simulations = 3

    with open("configs/time_test_1000.json") as f:
        cro_configs = json.load(f)

    X, y = load_dataset(data_config)
    M = vt.create_mask_dx(depth)
    n_attributes = data_config["n_attributes"]
    n_classes = data_config["n_classes"]
    n_samples = len(X)
    n_leaves = 2 ** depth

    for simulation in range(simulations):
        X_train = X
        y_train = y

        X_train_ = np.vstack((np.ones(len(X_train)).T, X_train.T)).T
        Y_train_ = np.tile(y_train, (2 ** depth, 1))

        if args["evaluation_scheme"] == "tensorflow":
            X_train_ = tf.convert_to_tensor(X_train_, dtype=tf.float64)
            Y_train_ = tf.convert_to_tensor(Y_train_, dtype=tf.int32)
            M = tf.convert_to_tensor(M, dtype=tf.int32)


        # Creating objective function based on execution parameters
        class SupervisedObjectiveFunc(AbsObjetiveFunc):
            def __init__(self, size, opt="max"):
                self.size = size
                self.time_test = True
                super().__init__(self.size, opt)

            def objetive(self, solution):
                if args["evaluation_scheme"] == "tensorflow":
                    W = tf.cast(tf.reshape(solution, [2 ** depth - 1, n_attributes + 1]), dtype=tf.float64)
                else:
                    W = solution.reshape((2 ** depth - 1, n_attributes + 1))

                accuracy, _ = fitness_evaluation(X_train, y_train, W, depth, n_classes, X_train_, Y_train_, M)
                return accuracy

            def random_solution(self):
                if args["evaluation_scheme"] == "tensorflow":
                    return tf.random.uniform(shape=[(2 ** depth - 1) * (n_attributes + 1)], minval=-1, maxval=1)
                else:
                    return vt.generate_random_weights(n_attributes, depth)

            def check_bounds(self, solution):
                if args["evaluation_scheme"] == "tensorflow":
                    return tf.clip_by_value(solution, -1, 1)
                else:
                    return np.clip(solution.copy(), -1, 1)


        sol_size = len(vt.generate_random_weights(n_attributes, depth).flatten())
        objfunc = SupervisedObjectiveFunc(sol_size)
        c = CRO_SL(objfunc, get_substrates_real(cro_configs), cro_configs["general"])

        start_time = time.perf_counter()
        for ev in range(1000):
            if ev % 100 == 0:
                print(f"... {ev} / 1000")
            W = np.random.uniform(-1, 1, (n_leaves - 1, n_attributes + 1))
            # W = vt.get_W_as_univariate(W)
            fitness_evaluation(X_train, y_train, W, depth, n_classes, X_train_, Y_train_, M)

        # c.optimize()
        # cProfile.run("c.optimize()", 'restats')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {end_time - start_time} seconds")