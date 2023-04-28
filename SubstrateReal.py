import random
import sys
sys.path.append(".")
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
