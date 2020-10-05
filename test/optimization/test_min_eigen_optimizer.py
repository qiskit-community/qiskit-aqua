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

""" Test Min Eigen Optimizer """

import unittest
from os import path
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from ddt import ddt, data
import numpy as np

from qiskit import BasicAer
from qiskit.aqua import MissingOptionalLibraryError
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.converters import (IntegerToBinary, InequalityToEquality,
                                            LinearEqualityToPenalty, QuadraticProgramToQubo)
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationResultStatus


@ddt
class TestMinEigenOptimizer(QiskitOptimizationTestCase):
    """Min Eigen Optimizer Tests."""

    def setUp(self):
        super().setUp()

        # setup minimum eigen solvers
        self.min_eigen_solvers = {}

        # exact eigen solver
        self.min_eigen_solvers['exact'] = NumPyMinimumEigensolver()

        # QAOA
        optimizer = COBYLA()
        self.min_eigen_solvers['qaoa'] = QAOA(optimizer=optimizer)

    @data(
        ('exact', None, 'op_ip1.lp'),
        ('qaoa', 'statevector_simulator', 'op_ip1.lp'),
        ('qaoa', 'qasm_simulator', 'op_ip1.lp')
    )
    def test_min_eigen_optimizer(self, config):
        """ Min Eigen Optimizer Test """
        try:
            # unpack configuration
            min_eigen_solver_name, backend, filename = config

            # get minimum eigen solver
            min_eigen_solver = self.min_eigen_solvers[min_eigen_solver_name]
            if backend:
                min_eigen_solver.quantum_instance = BasicAer.get_backend(backend)

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(path.join('resources', filename))
            problem.read_from_lp_file(lp_file)

            # solve problem with cplex
            cplex = CplexOptimizer()
            cplex_result = cplex.solve(problem)

            # solve problem
            result = min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result)

            # analyze results
            self.assertAlmostEqual(cplex_result.fval, result.fval)

            # check that eigensolver result is present
            self.assertIsNotNone(result.min_eigen_solver_result)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except RuntimeError as ex:
            self.fail(str(ex))

    @data(
        ('op_ip1.lp', -470, 12, OptimizationResultStatus.SUCCESS),
        ('op_ip1.lp', np.inf, None, OptimizationResultStatus.FAILURE),
        )
    def test_min_eigen_optimizer_with_filter(self, config):
        """ Min Eigen Optimizer Test """
        try:
            # unpack configuration
            filename, lowerbound, fval, status = config

            # get minimum eigen solver
            min_eigen_solver = NumPyMinimumEigensolver()

            # set filter
            # pylint: disable=unused-argument
            def filter_criterion(x, v, aux):
                return v > lowerbound
            min_eigen_solver.filter_criterion = filter_criterion

            # construct minimum eigen optimizer
            min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)

            # load optimization problem
            problem = QuadraticProgram()
            lp_file = self.get_resource_path(path.join('resources', filename))
            problem.read_from_lp_file(lp_file)

            # solve problem
            result = min_eigen_optimizer.solve(problem)
            self.assertIsNotNone(result)

            # analyze results
            self.assertAlmostEqual(fval, result.fval)
            self.assertAlmostEqual(status, result.status)

            # check that eigensolver result is present
            self.assertIsNotNone(result.min_eigen_solver_result)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except RuntimeError as ex:
            self.fail(str(ex))

    def test_converter_list(self):
        """Test converter list"""
        op = QuadraticProgram()
        op.integer_var(0, 3, "x")
        op.binary_var('y')

        op.maximize(linear={'x': 1, 'y': 2})
        op.linear_constraint(linear={'x': 1, 'y': 1}, sense='LE', rhs=3, name='xy_leq')
        min_eigen_solver = NumPyMinimumEigensolver()
        # a single converter
        qp2qubo = QuadraticProgramToQubo()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver, converters=qp2qubo)
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        # a list of converters
        ineq2eq = InequalityToEquality()
        int2bin = IntegerToBinary()
        penalize = LinearEqualityToPenalty()
        converters = [ineq2eq, int2bin, penalize]
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver, converters=converters)
        self.assertEqual(result.fval, 4)
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            MinimumEigenOptimizer(min_eigen_solver,
                                  converters=invalid)


if __name__ == '__main__':
    unittest.main()
