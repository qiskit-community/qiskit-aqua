# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
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

import numpy as np
from ddt import data, ddt

from qiskit import BasicAer
from qiskit.aqua import MissingOptionalLibraryError, QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.optimization.algorithms import (CplexOptimizer, MinimumEigenOptimizer)
from qiskit.optimization.algorithms.optimization_algorithm import OptimizationResultStatus
from qiskit.optimization.converters import (InequalityToEquality, IntegerToBinary,
                                            LinearEqualityToPenalty, QuadraticProgramToQubo)
from qiskit.optimization.problems import QuadraticProgram


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
            self.assertEqual(status, result.status)

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
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(result.fval, 4)
        with self.assertRaises(TypeError):
            invalid = [qp2qubo, "invalid converter"]
            MinimumEigenOptimizer(min_eigen_solver,
                                  converters=invalid)

    def test_samples(self):
        """Test samples"""
        SUCCESS = OptimizationResultStatus.SUCCESS  # pylint: disable=invalid-name
        aqua_globals.random_seed = 123
        quantum_instance = QuantumInstance(backend=BasicAer.get_backend('qasm_simulator'),
                                           seed_simulator=123, seed_transpiler=123,
                                           shots=1000)

        # test minimize
        op = QuadraticProgram()
        op.integer_var(0, 3, 'x')
        op.binary_var('y')
        op.minimize(linear={'x': 1, 'y': 2})
        op.linear_constraint(linear={'x': 1, 'y': 1}, sense='>=', rhs=1, name='xy')

        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        result = min_eigen_optimizer.solve(op)
        opt_sol = 1
        self.assertEqual(result.fval, opt_sol)
        self.assertEqual(len(result.samples), 1)
        np.testing.assert_array_almost_equal(result.samples[0].x, [1, 0])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.samples[0].probability, 1.0)
        self.assertEqual(result.samples[0].status, SUCCESS)
        self.assertEqual(len(result.raw_samples), 1)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [1, 0, 0, 0, 0])
        self.assertAlmostEqual(result.raw_samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.raw_samples[0].probability, 1.0)
        self.assertEqual(result.raw_samples[0].status, SUCCESS)

        qaoa = QAOA(quantum_instance=quantum_instance)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(len(result.samples), 8)
        self.assertEqual(len(result.raw_samples), 32)
        self.assertAlmostEqual(sum(s.probability for s in result.samples), 1)
        self.assertAlmostEqual(sum(s.probability for s in result.raw_samples), 1)
        self.assertAlmostEqual(min(s.fval for s in result.samples), 0)
        self.assertAlmostEqual(min(s.fval for s in result.samples if s.status == SUCCESS), opt_sol)
        self.assertAlmostEqual(min(s.fval for s in result.raw_samples), opt_sol)
        for sample in result.raw_samples:
            self.assertEqual(sample.status, SUCCESS)
        np.testing.assert_array_almost_equal(result.x, result.samples[0].x)
        self.assertAlmostEqual(result.fval, result.samples[0].fval)
        self.assertEqual(result.status, result.samples[0].status)

        # test maximize
        op = QuadraticProgram()
        op.integer_var(0, 3, 'x')
        op.binary_var('y')
        op.maximize(linear={'x': 1, 'y': 2})
        op.linear_constraint(linear={'x': 1, 'y': 1}, sense='<=', rhs=1, name='xy')

        min_eigen_solver = NumPyMinimumEigensolver()
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        result = min_eigen_optimizer.solve(op)
        opt_sol = 2
        self.assertEqual(result.fval, opt_sol)
        self.assertEqual(len(result.samples), 1)
        np.testing.assert_array_almost_equal(result.samples[0].x, [0, 1])
        self.assertAlmostEqual(result.samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.samples[0].probability, 1.0)
        self.assertEqual(result.samples[0].status, SUCCESS)
        self.assertEqual(len(result.raw_samples), 1)
        np.testing.assert_array_almost_equal(result.raw_samples[0].x, [0, 0, 1, 0])
        self.assertAlmostEqual(result.raw_samples[0].fval, opt_sol)
        self.assertAlmostEqual(result.raw_samples[0].probability, 1.0)
        self.assertEqual(result.raw_samples[0].status, SUCCESS)

        qaoa = QAOA(quantum_instance=quantum_instance)
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(op)
        self.assertEqual(len(result.samples), 8)
        self.assertEqual(len(result.raw_samples), 16)
        self.assertAlmostEqual(sum(s.probability for s in result.samples), 1)
        self.assertAlmostEqual(sum(s.probability for s in result.raw_samples), 1)
        self.assertAlmostEqual(max(s.fval for s in result.samples), 5)
        self.assertAlmostEqual(max(s.fval for s in result.samples if s.status == SUCCESS), opt_sol)
        self.assertAlmostEqual(max(s.fval for s in result.raw_samples), opt_sol)
        for sample in result.raw_samples:
            self.assertEqual(sample.status, SUCCESS)
        np.testing.assert_array_almost_equal(result.x, result.samples[0].x)
        self.assertAlmostEqual(result.fval, result.samples[0].fval)
        self.assertEqual(result.status, result.samples[0].status)


if __name__ == '__main__':
    unittest.main()
