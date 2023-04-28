import random

import numpy as np
import tensorflow as tf
from operators import *
from Substrate import *

"""
Substrate class that has continuous mutation and cross methods
"""


class SubstrateReal(Substrate):
    def __init__(self, evolution_method, params=None):
        self.evolution_method = evolution_method
        super().__init__(self.evolution_method, params)

    """
    Evolves a solution with a different strategy depending on the type of operator
    """

    def evolve(self, solution, population, objfunc):
        others = [i for i in population if i != solution]

        if objfunc.algo.startswith("tf"):
            fitnesses = [i.fitness for i in population]
            best = population[np.argmax(fitnesses)]
            result = solution.solution

            if self.evolution_method == "DE/rand/1":
                if len(population) > 3:
                    r1, r2, r3 = random.sample(population, 3)
                    result = DERand1_tf(solution.solution, r1, r2, r3, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/best/1":
                if len(population) > 2:
                    r1, r2 = random.sample(population, 2)
                    result = DEBest1_tf(solution.solution, r1, r2, best, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/rand/2":
                if len(population) > 5:
                    r1, r2, r3, r4, r5 = random.sample(population, 5)
                    result = DERand2_tf(solution.solution, r1, r2, r3, r4, r5, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/best/2":
                if len(population) > 4:
                    r1, r2, r3, r4 = random.sample(population, 4)
                    result = DEBest2_tf(solution.solution, r1, r2, r3, r4, best, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-rand/1":
                if len(population) > 3:
                    r1, r2, r3 = random.sample(population, 3)
                    result = DECurrentToRand1_tf(solution.solution, r1, r2, r3, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-best/1":
                if len(population) > 2:
                    r1, r2 = random.sample(population, 2)
                    result = DECurrentToBest1_tf(solution.solution, r1, r2, best, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-pbest/1":
                if len(population) > 2:
                    p = 0.11
                    r1, r2 = random.sample(population, 2)
                    pbest_idx = random.choice(np.argsort(fitnesses)[:math.ceil(len(population) * p)])
                    pbest = population[pbest_idx]
                    result = DECurrentToPBest1_tf(solution.solution, r1, r2, pbest, self.params["F"], self.params["Cr"])
            else:
                print(f"Error: evolution method \"{self.evolution_method}\" not defined")
                exit(1)
        else:
            sol = solution.solution.copy()

            if self.evolution_method == "DE/rand/1":
                result = DERand1(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/best/1":
                result = DEBest1(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/rand/2":
                result = DERand2(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/best/2":
                result = DEBest2(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-rand/1":
                result = DECurrentToRand1(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-best/1":
                result = DECurrentToBest1(sol, others, self.params["F"], self.params["Cr"])
            elif self.evolution_method == "DE/current-to-pbest/1":
                result = DECurrentToPBest1(sol, others, self.params["F"], self.params["Cr"])
            else:
                print(f"Error: evolution method \"{self.evolution_method}\" not defined")
                exit(1)

        return result
