# -*- coding: utf-8 -*-

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

from qiskit import BasicAer

from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import COBYLA

from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
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

            # analyze results
            self.assertAlmostEqual(cplex_result.fval, result.fval)
        except RuntimeError as ex:
            msg = str(ex)
            if 'CPLEX' in msg:
                self.skipTest(msg)
            else:
                self.fail(msg)


if __name__ == '__main__':
    unittest.main()
