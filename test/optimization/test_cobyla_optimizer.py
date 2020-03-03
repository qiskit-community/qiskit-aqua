# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Cobyla Optimizer """

from test.optimization.common import QiskitOptimizationTestCase
import numpy as np

from qiskit.optimization.algorithms import CobylaOptimizer
from qiskit.optimization.problems import OptimizationProblem

from ddt import ddt, data


@ddt
class TestCobylaOptimizer(QiskitOptimizationTestCase):
    """Cobyla Optimizer Tests."""

    def setUp(self):
        super().setUp()

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
        problem = OptimizationProblem()
        problem.read(self.resource_path + filename)

        # solve problem with cobyla
        result = self.cobyla_optimizer.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, fval)
