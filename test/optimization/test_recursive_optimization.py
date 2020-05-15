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

"""Test Recursive Min Eigen Optimizer."""

import unittest
from os import path
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

from qiskit.aqua.algorithms import NumPyMinimumEigensolver

from qiskit.optimization.algorithms import (MinimumEigenOptimizer, CplexOptimizer,
                                            RecursiveMinimumEigenOptimizer)
from qiskit.optimization.problems import QuadraticProgram


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
        except RuntimeError as ex:
            msg = str(ex)
            if 'CPLEX' in msg:
                self.skipTest(msg)
            else:
                self.fail(msg)


if __name__ == '__main__':
    unittest.main()
