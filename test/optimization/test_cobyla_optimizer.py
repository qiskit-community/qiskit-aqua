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

from qiskit.optimization.algorithms import CobylaOptimizer
from qiskit.optimization.problems import QuadraticProgram

logger = logging.getLogger(__name__)


class TestCobylaOptimizer(QiskitOptimizationTestCase):
    """Cobyla Optimizer Tests. """

    def test_cobyla_optimizer(self):
        """ Cobyla Optimizer Test. """

        # load optimization problem
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense='=', rhs=2)
        problem.minimize(linear=[2, 2], quadratic=[[2, 0.25], [0.25, 0.5]])

        # solve problem with cobyla
        cobyla = CobylaOptimizer()
        result = cobyla.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, 5.8750)

    def test_cobyla_optimizer_with_quadratic_constraint(self):
        """ Cobyla Optimizer Test With Quadratic Constraints. """
        # load optimization problem
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=1)
        problem.continuous_var(upperbound=1)

        problem.minimize(linear=[1, 1])

        linear = [-1, -1]
        quadratic = [[1, 0], [0, 1]]
        problem.quadratic_constraint(linear=linear, quadratic=quadratic, rhs=-1/2)

        # solve problem with cobyla
        cobyla = CobylaOptimizer()
        result = cobyla.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, 1.0, places=2)

    def test_cobyla_optimizer_with_variable_bounds(self):
        """ Cobyla Optimizer Test With Variable Bounds. """

        # initialize optimizer
        cobyla = CobylaOptimizer()

        # initialize problem
        problem = QuadraticProgram()

        # set variables and bounds
        problem.continuous_var(lowerbound=-1, upperbound=1)
        problem.continuous_var(lowerbound=-2, upperbound=2)

        # set objective and minimize
        problem.minimize(linear=[1, 1])

        # solve problem with cobyla
        result = cobyla.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.x[0], -1.0, places=6)
        self.assertAlmostEqual(result.x[1], -2.0, places=6)

        # set objective and minimize
        problem.maximize(linear=[1, 1])

        # solve problem with cobyla
        result = cobyla.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.x[0], 1.0, places=6)
        self.assertAlmostEqual(result.x[1], 2.0, places=6)

    def test_cobyla_optimizer_with_trials(self):
        """ Cobyla Optimizer Test. """

        # load optimization problem
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense='=', rhs=2)
        problem.minimize(linear=[2, 2], quadratic=[[2, 0.25], [0.25, 0.5]])

        # solve problem with cobyla
        cobyla = CobylaOptimizer(trials=3)
        result = cobyla.solve(problem)

        # analyze results
        self.assertAlmostEqual(result.fval, 5.8750)


if __name__ == '__main__':
    unittest.main()
