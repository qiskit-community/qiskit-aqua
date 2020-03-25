# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests of the ADMM algorithm."""
from typing import Optional

from test.optimization import QiskitOptimizationTestCase

import numpy as np
from cplex import SparsePair
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit.optimization.algorithms.admm_optimizer import ADMMOptimizer, ADMMParameters
from qiskit.optimization.problems import OptimizationProblem


class TestADMMOptimizerMiskp(QiskitOptimizationTestCase):
    """ADMM Optimizer Tests based on Mixed-Integer Setup Knapsack Problem"""

    def test_admm_optimizer_miskp_eigen(self):
        """ADMM Optimizer Test based on Mixed-Integer Setup Knapsack Problem
        using NumPy eigen optimizer"""
        miskp = Miskp(*self._get_problem_params())
        op: OptimizationProblem = miskp.create_problem()
        self.assertIsNotNone(op)

        # use numpy exact diagonalization
        qubo_optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        continuous_optimizer = CplexOptimizer()

        admm_params = ADMMParameters(qubo_optimizer=qubo_optimizer,
                                     continuous_optimizer=continuous_optimizer)

        solver = ADMMOptimizer(params=admm_params)
        solution = solver.solve(op)
        self.assertIsNotNone(solution)

        correct_solution = [0.009127, 0.009127, 0.009127, 0.009127, 0.009127, 0.009127, 0.009127,
                            0.009127, 0.009127, 0.009127, 0.006151, 0.006151, 0.006151, 0.006151,
                            0.006151, 0.006151, 0.006151, 0.006151, 0.006151, 0.006151,
                            0., 0.]
        correct_objective = -1.2113693

        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal(correct_solution, solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(correct_objective, solution.fval, 3)

    @staticmethod
    def _get_problem_params() -> (int, int, float, np.ndarray, np.ndarray, np.ndarray):
        """Fills in parameters for a Mixed Integer Setup Knapsack Problem (MISKP) instance."""

        family_count = 2
        items_per_family = 10
        knapsack_capacity = 45.10
        setup_costs = np.asarray([75.61, 75.54])

        resource_values = np.asarray([9.78, 8.81, 9.08, 2.03, 8.9, 9, 4.12, 8.16, 6.55, 4.84, 3.78,
                                      5.13, 4.72, 7.6, 2.83, 1.44, 2.45, 2.24, 6.3, 5.02]) \
            .reshape((family_count, items_per_family))

        cost_values = np.asarray([-11.78, -10.81, -11.08, -4.03, -10.90, -11.00, -6.12, -10.16,
                                  -8.55, -6.84, -5.78, -7.13, -6.72, -9.60, -4.83, -3.44, -4.45,
                                  -4.24, -8.30, -7.02]) \
            .reshape((family_count, items_per_family))

        return family_count, items_per_family, knapsack_capacity, setup_costs, \
               resource_values, cost_values


class Miskp:
    """A Helper class to generate  Mixed Integer Setup Knapsack problems"""
    def __init__(self, family_count: int, items_per_family: int, knapsack_capacity: float,
                 setup_costs: np.ndarray, resource_values: np.ndarray, cost_values: np.ndarray)\
            -> None:
        """Constructs an instance of this helper class to create suitable ADMM problems.

        Args:
            family_count: number of families
            items_per_family: number of items in each family
            knapsack_capacity: capacity of the knapsack
            setup_costs: setup cost to include family k in the knapsack
            resource_values: resources consumed if item t in family k is included in the knapsack
            cost_values: value of including item t in family k in the knapsack
        """

        self.knapsack_capacity = knapsack_capacity
        self.setup_costs = setup_costs
        self.resource_values = resource_values
        self.cost_values = cost_values
        self.items_per_family = items_per_family
        self.family_count = family_count

        # definitions of the internal variables
        self.op = None
        self.range_family = None
        self.range_items = None
        self.n_x_vars = None
        self.n_y_vars = None
        self.range_x_vars = None
        self.range_y_vars = None

    @staticmethod
    def _var_name(stem: str, index1: int, index2: Optional[int] = None,
                  index3: Optional[int] = None) -> str:
        """A method to return a string representation of the name of a decision variable or
        a constraint, given its indices.

        Args:
            stem: Element name.
            index1: Element indices
            index2: Element indices
            index3: Element indices

        Returns:
            Textual representation of the variable name based on the parameters
        """
        if index2 is None:
            return stem + "(" + str(index1) + ")"
        if index3 is None:
            return stem + "(" + str(index1) + "," + str(index2) + ")"
        return stem + "(" + str(index1) + "," + str(index2) + "," + str(index3) + ")"

    def _create_params(self) -> None:
        self.range_family = range(self.family_count)
        self.range_items = range(self.items_per_family)

        # make sure instance params are floats
        self.setup_costs = [float(val) for val in self.setup_costs]
        self.cost_values = self.cost_values.astype(float)
        self.resource_values = self.resource_values.astype(float)

        self.n_x_vars = self.family_count * self.items_per_family
        self.n_y_vars = self.family_count

        self.range_x_vars = [(k, t) for k in self.range_family for t in self.range_items]
        self.range_y_vars = self.range_family

    def _create_vars(self) -> None:
        self.op.variables.add(
            lb=[0.0] * self.n_x_vars,
            names=[self._var_name("x", i, j) for i, j in self.range_x_vars])

        self.op.variables.add(
            # lb=[0.0] * self.n_y_vars,
            # ub=[1.0] * self.n_y_vars,
            types=["B"] * self.n_y_vars,
            names=[self._var_name("y", i) for i in self.range_y_vars])

    def _create_constraint_capacity(self) -> None:
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[self._var_name("x", i, j) for i, j in self.range_x_vars]
                    ,
                    val=[self.resource_values[i, j] for i, j in self.range_x_vars])
            ],
            senses="L",
            rhs=[self.knapsack_capacity],
            names=["CAPACITY"])

    def _create_constraint_allocation(self) -> None:
        self.op.linear_constraints.add(
            lin_expr=[
                SparsePair(
                    ind=[self._var_name("x", k, t)] + [self._var_name("y", k)],
                    val=[1.0, -1.0])
                for k, t in self.range_x_vars
            ],
            senses="L" * self.n_x_vars,
            rhs=[0.0] * self.n_x_vars,
            names=[self._var_name("ALLOCATION", k, t) for k, t in self.range_x_vars])

    def _create_constraints(self) -> None:
        self._create_constraint_capacity()
        self._create_constraint_allocation()

    def _create_objective(self) -> None:
        self.op.objective.set_linear([(self._var_name("y", k), self.setup_costs[k])
                                      for k in self.range_family] +
                                     [(self._var_name("x", k, t), self.cost_values[k, t])
                                      for k, t in
                                      self.range_x_vars]
                                     )

    def create_problem(self) -> OptimizationProblem:
        """Creates an instance of optimization problem based on parameters specified.

        Returns:
            an instance of optimization problem.
        """
        self.op = OptimizationProblem()

        self._create_params()
        self._create_vars()
        self._create_objective()
        self._create_constraints()

        return self.op
