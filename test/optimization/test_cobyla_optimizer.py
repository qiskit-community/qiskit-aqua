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

""" Test Cobyla Optimizer """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import logging
from ddt import ddt, data

from qiskit.optimization.algorithms import CobylaOptimizer
from qiskit.optimization.problems import QuadraticProgram

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import SparsePair, SparseTriple
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


@ddt
class TestCobylaOptimizer(QiskitOptimizationTestCase):
    """Cobyla Optimizer Tests."""

    def setUp(self):
        super().setUp()
        if not _HAS_CPLEX:
            self.skipTest('CPLEX is not installed.')

        self.resource_path = './test/optimization/resources/'
        self.cobyla_optimizer = CobylaOptimizer()

    @data(
        ('op_lp1.lp', 5.8750)
    )
    def test_cobyla_optimizer(self, config):
        """ Cobyla Optimizer Test """

        # unpack configuration
        filename, fval = config

        # load optimization problem
        problem = QuadraticProgram()
        problem.read(self.resource_path + filename)

        # solve problem with cobyla
        result = self.cobyla_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, fval)

    def test_cobyla_optimizer_with_quadratic_constraint(self):
        """ Cobyla Optimizer Test """
        # load optimization problem
        problem = QuadraticProgram()
        problem.variables.add(lb=[0, 0], ub=[1, 1], types='CC')
        problem.objective.set_linear([(0, 1), (1, 1)])

        qc = problem.quadratic_constraints
        linear = SparsePair(ind=[0, 1], val=[-1, -1])
        quadratic = SparseTriple(ind1=[0, 1], ind2=[0, 1], val=[1, 1])
        qc.add(name='qc', lin_expr=linear, quad_expr=quadratic, rhs=-1/2)

        # solve problem with cobyla
        result = self.cobyla_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
