# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test SLSQP Optimizer """
import unittest

from test.optimization.optimization_test_case import QiskitOptimizationTestCase

import numpy as np

from qiskit.optimization import INFINITY
from qiskit.optimization.algorithms import SlsqpOptimizer
from qiskit.optimization.problems import QuadraticProgram


class TestSlsqpOptimizer(QiskitOptimizationTestCase):
    """SLSQP Optimizer Tests. """

    def test_slsqp_optimizer(self):
        """ Generic SLSQP Optimizer Test. """

        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense='=', rhs=2)
        problem.minimize(linear=[2, 2], quadratic=[[2, 0.25], [0.25, 0.5]])

        # solve problem with SLSQP
        slsqp = SlsqpOptimizer(trials=3)
        result = slsqp.solve(problem)

        self.assertAlmostEqual(result.fval, 5.8750)

    def test_slsqp_optimizer_full_output(self):
        """ Generic SLSQP Optimizer Test. """

        problem = QuadraticProgram()
        problem.continuous_var(upperbound=4)
        problem.continuous_var(upperbound=4)
        problem.linear_constraint(linear=[1, 1], sense='=', rhs=2)
        problem.minimize(linear=[2, 2], quadratic=[[2, 0.25], [0.25, 0.5]])

        # solve problem with SLSQP
        slsqp = SlsqpOptimizer(trials=3, full_output=True)
        result = slsqp.solve(problem)

        self.assertAlmostEqual(result.fval, 5.8750)

        self.assertAlmostEqual(result.fx, 5.8750)
        self.assertGreaterEqual(result.its, 1)
        self.assertEqual(result.imode, 0)
        self.assertIsNotNone(result.smode)
        self.assertEqual(len(result.samples), 1)
        self.assertAlmostEqual(result.fval, result.samples[0].fval)
        np.testing.assert_almost_equal(result.x, result.samples[0].x)
        self.assertEqual(result.status, result.samples[0].status)
        self.assertAlmostEqual(result.samples[0].probability, 1.0)

    def test_slsqp_unbounded(self):
        """Unbounded test for optimization"""
        problem = QuadraticProgram()
        problem.continuous_var(name="x")
        problem.continuous_var(name="y")
        problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

        slsqp = SlsqpOptimizer()
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([2., 1.], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(2., solution.fval, 3)

    def test_slsqp_unbounded_with_trials(self):
        """Unbounded test for optimization"""
        problem = QuadraticProgram()
        problem.continuous_var(name="x", lowerbound=-INFINITY, upperbound=INFINITY)
        problem.continuous_var(name="y", lowerbound=-INFINITY, upperbound=INFINITY)
        problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

        slsqp = SlsqpOptimizer(trials=3)
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([2., 1.], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(2., solution.fval, 3)

    def test_slsqp_bounded(self):
        """Same as above, but a bounded test"""
        problem = QuadraticProgram()
        problem.continuous_var(name="x", lowerbound=2.5)
        problem.continuous_var(name="y", upperbound=0.5)
        problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

        slsqp = SlsqpOptimizer()
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([2.5, 0.5], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(0.75, solution.fval, 3)

    def test_slsqp_equality(self):
        """A test with equality constraint"""
        problem = QuadraticProgram()
        problem.continuous_var(name="x")
        problem.continuous_var(name="y")
        problem.linear_constraint(linear=[1, -1], sense='=', rhs=0)
        problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

        slsqp = SlsqpOptimizer()
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([1., 1.], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(1., solution.fval, 3)

    def test_slsqp_inequality(self):
        """A test with inequality constraint"""
        problem = QuadraticProgram()
        problem.continuous_var(name="x")
        problem.continuous_var(name="y")
        problem.linear_constraint(linear=[1, -1], sense='>=', rhs=1)
        problem.maximize(linear=[2, 0], quadratic=[[-1, 2], [0, -2]])

        slsqp = SlsqpOptimizer()
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([2., 1.], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(2., solution.fval, 3)

    def test_slsqp_optimizer_with_quadratic_constraint(self):
        """A test with equality constraint"""
        problem = QuadraticProgram()
        problem.continuous_var(upperbound=1)
        problem.continuous_var(upperbound=1)

        problem.minimize(linear=[1, 1])

        linear = [-1, -1]
        quadratic = [[1, 0], [0, 1]]
        problem.quadratic_constraint(linear=linear, quadratic=quadratic, rhs=-1 / 2)

        slsqp = SlsqpOptimizer()
        solution = slsqp.solve(problem)

        self.assertIsNotNone(solution)
        self.assertIsNotNone(solution.x)
        np.testing.assert_almost_equal([0.5, 0.5], solution.x, 3)
        self.assertIsNotNone(solution.fval)
        np.testing.assert_almost_equal(1., solution.fval, 3)

    def test_multistart_properties(self):
        """
        Tests properties of MultiStartOptimizer.
        Since it is an abstract class, the test is here.
        """
        trials = 5
        clip = 200.

        slsqp = SlsqpOptimizer(trials=trials, clip=clip)
        self.assertEqual(trials, slsqp.trials)
        self.assertAlmostEqual(clip, slsqp.clip)

        trials = 6
        clip = 300.
        slsqp.trials = trials
        slsqp.clip = clip
        self.assertEqual(trials, slsqp.trials)
        self.assertAlmostEqual(clip, slsqp.clip)


if __name__ == '__main__':
    unittest.main()
