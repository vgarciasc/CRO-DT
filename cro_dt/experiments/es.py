import time

from cro_dt.sup_configs import get_config
from cro_dt.experiments.es_tree import Tree
from cro_dt.experiments.es_tree2 import TreeFlat
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib as jl

from rich.progress import track, Progress


def evaluate_population(X_train, y_train, population):
    for individual in population:
        individual.update_labels(X_train, y_train)
        individual.fitness = individual.evaluate(X_train, y_train)
    return population

def run_es(dataset, X_train, y_train, X_test, y_test,
           lamb, mu, n_gens, depth, simulation_id=0,
           max_gens_wout_improvement=100, n_jobs=1):
    config = get_config(dataset)
    config["attributes_metadata"] = [(np.min(X_i), np.max(X_i)) for X_i in np.transpose(X_train.astype(np.float32))]

    # population = [Tree.generate_random_tree(config, depth) for _ in range(lamb)]
    population = [TreeFlat.generate_random(config, depth) for _ in range(lamb)]
    for individual in population:
        individual.update_labels(X_train, y_train)
        individual.fitness = individual.evaluate(X_train, y_train)

    last_improvement_gen_id = 0
    best_fitness = -1
    best_tree = None

    with Progress() as progress:
        task = progress.add_task("[bold green]Running ES...", total=n_gens)

        for curr_gen in range(n_gens):
            if (curr_gen - last_improvement_gen_id) > max_gens_wout_improvement:
                print(
                    f"Stopping early at generation {curr_gen} (no improvement for {max_gens_wout_improvement} generations.")
                break

            population_par = jl.parallel.Parallel(n_jobs=n_jobs)(
                jl.parallel.delayed(evaluate_population)(X_train, y_train, population[i::n_jobs]) for i in range(n_jobs))
            population = [ind for sublist in population_par for ind in sublist]

            for individual in population:
                if individual.fitness > best_fitness:
                    best_fitness = individual.fitness
                    best_tree = individual
                    last_improvement_gen_id = curr_gen

            population.sort(key=lambda x: x.fitness, reverse=True)
            parents = population[:mu]
            child_population = []

            for parent in parents:
                for _ in range(lamb // mu):
                    child = parent.copy()
                    child.mutate()
                    child_population.append(child)

            population = parents + child_population

            progress.update(task, advance=1, description=f"Running ES... Simulation {simulation_id} // "
                                                         f"Generation {curr_gen} "
                                                         f"(last improv. {last_improvement_gen_id}) // "
                                                         f"Best fitness: {'{:.5f}'.format(best_fitness)}")

    print("\nBest tree:\n")
    print(best_tree)
    print(f"Best tree in-sample accuracy: {best_tree.fitness}")
    print(f"Best tree out-of-sample accuracy: {best_tree.evaluate(X_test, y_test)}")

    return best_tree


if __name__ == "__main__":
    lamb = 100
    mu = 10
    n_generations = 30
    depth = 2
    dataset = "breast_cancer"

    df = pd.read_csv(f"cro_dt/experiments/data/{dataset}.csv")
    feat_names = df.columns[:-1]
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X = X.astype(np.float64)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=2, stratify=y_test)

    run_es(dataset, X_train, y_train, X_test, y_test, lamb, mu, n_generations, depth)
