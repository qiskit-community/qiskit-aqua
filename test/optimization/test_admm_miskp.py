# -*-coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2000.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the ADMM algorithm.
"""

import numpy as np
from cplex import SparsePair
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMOptimizer, ADMMParameters
from qiskit.optimization.problems import OptimizationProblem

from test.optimization import QiskitOptimizationTestCase


class TestADMMOptimizerMiskp(QiskitOptimizationTestCase):
    """ADMM Optimizer Tests based on Mixed-Integer Setup Knapsack Problem"""

    def setUp(self):
        super().setUp()

    def test_admm_optimizer_miskp_eigen(self):
        """ ADMM Optimizer Test based on Mixed-Integer Setup Knapsack Problem using NumPy eigen optimizer"""
        K, T, P, S, D, C = self.get_problem_params()
        miskp = Miskp(K, T, P, S, D, C)
        op: OptimizationProblem = miskp.create_problem()

        # use numpy exact diagonalization
        qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        continuous_optimizer = CplexOptimizer()

        admm_params = ADMMParameters(qubo_optimizer=qubo_optimizer, continuous_optimizer=continuous_optimizer)

        solver = ADMMOptimizer(params=admm_params)
        solution = solver.solve(op)

        # debug
        print("results")
        print("x={}".format(solution.x))
        print("fval={}".format(solution.fval))

        correct_solution = [0.009127, 0.009127, 0.009127, 0.009127, 0.009127, 0.009127, 0.009127, 0.009127,
                            0.009127, 0.009127, 0.006151, 0.006151, 0.006151, 0.006151, 0.006151, 0.006151,
                            0.006151, 0.006151, 0.006151, 0.006151, 0.,       0.]
        correct_objective = -1.2113693

        np.testing.assert_almost_equal(correct_solution, solution.x, 3)
        np.testing.assert_almost_equal(solution.fval, correct_objective, 3)

    def get_problem_params(self):
        """
        Fills in parameters for a Mixed Integer Setup Knapsack Problem (MISKP) instance.
        """
        #

        K = 2
        T = 10
        P = 45.10
        S = np.asarray([75.61, 75.54])

        D = np.asarray([9.78, 8.81, 9.08, 2.03, 8.9, 9, 4.12, 8.16, 6.55, 4.84, 3.78, 5.13,
                        4.72, 7.6, 2.83, 1.44, 2.45, 2.24, 6.3, 5.02]).reshape((K, T))

        C = np.asarray([-11.78, -10.81, -11.08, -4.03, -10.90, -11.00, -6.12, -10.16, -8.55, -6.84,
                        -5.78, -7.13, -6.72, -9.60, -4.83, -3.44, -4.45, -4.24, -8.30, -7.02]).reshape((K, T))

        return K, T, P, S, D, C


class Miskp:
    def __init__(self, K, T, P, S, D: np.ndarray, C: np.ndarray, pairwise_incomp=0, multiple_choice=0):
        """
        Constructor method of the class.

        Args:
            K: number of families
            T: number of items in each family
            C: value of including item t in family k in the knapsack
            D: resources consumed if item t in family k is included in the knapsack
            S: setup cost to include family k in the knapsack
            P: capacity of the knapsack
        """

        self.multiple_choice = multiple_choice
        self.pairwise_incomp = pairwise_incomp
        self.P = P
        self.S = S
        self.D = D
        self.C = C
        self.T = T
        self.K = K

        # definitions of the internal variables
        self.op = None
        self.range_K = None
        self.range_T = None
        self.n_x_vars = None
        self.n_y_vars = None
        self.range_x_vars = None
        self.range_y_vars = None

    @staticmethod
    def var_name(stem, index1, index2=None, index3=None):
        """A method to return a string representing the name of a decision variable or a constraint, given its indices.
            Args:
                stem: Element name.
                index1: Element indices
                index2: Element indices
                index3: Element indices
        """
        if index2 is None:
            return stem + "(" + str(index1) + ")"
        if index3 is None:
            return stem + "(" + str(index1) + "," + str(index2) + ")"
        return stem + "(" + str(index1) + "," + str(index2) + "," + str(index3) + ")"

    def create_params(self):
        self.range_K = range(self.K)
        self.range_T = range(self.T)

        # make sure instance params are floats

        self.S = [float(val) for val in self.S]
        self.C = self.C.astype(float)
        self.D = self.D.astype(float)

        self.n_x_vars = self.K * self.T
        self.n_y_vars = self.K

        self.range_x_vars = [(k, t) for k in self.range_K for t in self.range_T]
        self.range_y_vars = self.range_K

    def create_vars(self):
        self.op.variables.add(
            lb=[0.0] * self.n_x_vars,
            names=[self.var_name("x", i, j) for i, j in self.range_x_vars])

        self.op.variables.add(
            # lb=[0.0] * self.n_y_vars,
            # ub=[1.0] * self.n_y_vars,
            types=["B"] * self.n_y_vars,
            names=[self.var_name("y", i) for i in self.range_y_vars])

    def create_constraint_capacity(self):
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[self.var_name("x", i, j) for i, j in self.range_x_vars]
                    ,
                    val=[self.D[i, j] for i, j in self.range_x_vars])
            ],
            senses="L",
            rhs=[self.P],
            names=["CAPACITY"])

    def create_constraint_allocation(self):
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[self.var_name("x", k, t)] + [self.var_name("y", k)],
                    val=[1.0, -1.0])
                for k, t in self.range_x_vars
            ],
            senses="L" * self.n_x_vars,
            rhs=[0.0] * self.n_x_vars,
            names=[self.var_name("ALLOCATION", k, t) for k, t in self.range_x_vars])

    def create_constraints(self):
        self.create_constraint_capacity()
        self.create_constraint_allocation()

    def create_objective(self):
        self.op.objective.set_linear([(self.var_name("y", k), self.S[k]) for k in self.range_K] +
                                     [(self.var_name("x", k, t), self.C[k, t]) for k, t in self.range_x_vars]
                                     )

    def create_problem(self):
        self.op = OptimizationProblem()

        self.create_params()
        self.create_vars()
        self.create_objective()
        self.create_constraints()

        return self.op
