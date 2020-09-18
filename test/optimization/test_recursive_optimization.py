# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Recursive Min Eigen Optimizer."""

import unittest
from os import path

from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from qiskit.optimization.algorithms.recursive_minimum_eigen_optimizer import IntermediateResult

from qiskit.aqua import MissingOptionalLibraryError
from qiskit.aqua.algorithms import NumPyMinimumEigensolver

from qiskit.optimization.algorithms import (MinimumEigenOptimizer, CplexOptimizer,
                                            RecursiveMinimumEigenOptimizer)
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import (IntegerToBinary, InequalityToEquality,
                                            LinearEqualityToPenalty, QuadraticProgramToQubo)


class TestRecursiveMinEigenOptimizer(QiskitOptimizationTestCase):
    """Recursive Min Eigen Optimizer Tests."""

    def test_recursive_min_eigen_optimizer(self):
        """Test the recursive minimum eigen optimizer."""
        try:
            filename = 'op_ip1.lp'
            # get minimum eigen solver
            min_eigen_solver = NumPyMinimumEigensolver()

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
            recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                                                           min_num_vars=4)

            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(path.join('resources', filename))
            problem.read_from_lp_file(lp_file)

            # solve problem with cplex
            cplex = CplexOptimizer()
            cplex_result = cplex.solve(problem)

            # solve problem
            result = recursive_min_eigen_optimizer.solve(problem)

            # analyze results
            self.assertAlmostEqual(cplex_result.fval, result.fval)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_min_eigen_optimizer_history(self):
        """Tests different options for history."""
        try:
            filename = 'op_ip1.lp'
            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(path.join('resources', filename))
            problem.read_from_lp_file(lp_file)

            # get minimum eigen solver
            min_eigen_solver = NumPyMinimumEigensolver()

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

            # no history
            recursive_min_eigen_optimizer = \
                RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                               min_num_vars=4,
                                               history=IntermediateResult.NO_ITERATIONS)
            result = recursive_min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result.replacements)
            self.assertIsNotNone(result.history)
            self.assertIsNotNone(result.history[0])
            self.assertEqual(len(result.history[0]), 0)
            self.assertIsNone(result.history[1])

            # only last iteration in the history
            recursive_min_eigen_optimizer = \
                RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                               min_num_vars=4,
                                               history=IntermediateResult.LAST_ITERATION)
            result = recursive_min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result.replacements)
            self.assertIsNotNone(result.history)
            self.assertIsNotNone(result.history[0])
            self.assertEqual(len(result.history[0]), 0)
            self.assertIsNotNone(result.history[1])

            # full history
            recursive_min_eigen_optimizer = \
                RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                               min_num_vars=4,
                                               history=IntermediateResult.ALL_ITERATIONS)
            result = recursive_min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result.replacements)
            self.assertIsNotNone(result.history)
            self.assertIsNotNone(result.history[0])
            self.assertGreater(len(result.history[0]), 1)
            self.assertIsNotNone(result.history[1])

        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_converter_list(self):
        """Test converter list"""
        op = QuadraticProgram()
        op.integer_var(0, 3, "x")
        op.binary_var('y')

        op.maximize(linear={'x': 1, 'y': 2})
        op.linear_constraint(linear={'y': 1, 'x': 1}, sense='LE', rhs=3, name='xy_leq')

        # construct minimum eigen optimizer
        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        # a single converter
        qp2qubo = QuadraticProgramToQubo()
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                                                       min_num_vars=2,
                                                                       converters=qp2qubo)
        result = recursive_min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # a list of converters
        ineq2eq = InequalityToEquality()
        int2bin = IntegerToBinary()
        penalize = LinearEqualityToPenalty()
        converters = [ineq2eq, int2bin, penalize]
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                                                       min_num_vars=2,
                                                                       converters=converters)
        result = recursive_min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # invalid converters
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                           min_num_vars=2,
                                           converters=invalid)


if __name__ == '__main__':
    unittest.main()
