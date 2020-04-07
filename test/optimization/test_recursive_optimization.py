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

""" Test Recursive Min Eigen Optimizer """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import numpy as np
from ddt import ddt, data, unpack

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.aqua.components.optimizers import COBYLA

from qiskit.optimization.algorithms import (MinimumEigenOptimizer, CplexOptimizer,
                                            RecursiveMinimumEigenOptimizer)
from qiskit.optimization.problems import OptimizationProblem


@ddt
class TestRecursiveMinEigenOptimizer(QiskitOptimizationTestCase):
    """Recursive Min Eigen Optimizer Tests."""

    def setUp(self):
        super().setUp()

        # fix random seed for reproducible results
        np.random.seed = 109
        aqua_globals.random_seed = 89

        self.resource_path = './test/optimization/resources/'

        # setup simulators
        self.qinstances = {}
        self.qinstances['qasm'] = QuantumInstance(
            BasicAer.get_backend('qasm_simulator'),
            shots=10000,
            seed_simulator=51,
            seed_transpiler=80
        )
        self.qinstances['statevector'] = QuantumInstance(
            BasicAer.get_backend('statevector_simulator'),
            seed_simulator=51,
            seed_transpiler=80
        )

        # setup minimum eigen solvers
        self.min_eigen_solvers = {}
        self.min_eigen_solvers['exact'] = NumPyMinimumEigensolver()
        self.min_eigen_solvers['qaoa'] = QAOA(optimizer=COBYLA())

    @data(
        ('exact', None, 'op_ip1.lp'),
        ('qaoa', 'statevector', 'op_ip1.lp'),
        ('qaoa', 'qasm', 'op_ip1.lp')
    )
    @unpack
    def test_recursive_min_eigen_optimizer(self, solver, simulator, filename):
        """ Min Eigen Optimizer Test """

        # get minimum eigen solver
        min_eigen_solver = self.min_eigen_solvers[solver]
        if simulator:
            min_eigen_solver.quantum_instance = self.qinstances[simulator]

        # construct minimum eigen optimizer
        min_eigen_optimizer = MinimumEigenOptimizer(min_eigen_solver)
        recursive_min_eigen_optimizer = RecursiveMinimumEigenOptimizer(min_eigen_optimizer,
                                                                       min_num_vars=4)

        # load optimization problem
        problem = OptimizationProblem()
        problem.read(self.resource_path + filename)

        # solve problem with cplex
        cplex = CplexOptimizer()
        cplex_result = cplex.solve(problem)

        # solve problem
        result = recursive_min_eigen_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(cplex_result.fval, result.fval)


if __name__ == '__main__':
    unittest.main()
