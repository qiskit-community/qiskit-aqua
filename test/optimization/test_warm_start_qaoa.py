# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test warm start QAOA optimizer. """
from test.optimization import QiskitOptimizationTestCase

import numpy as np
from docplex.mp.model import Model
from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import SlsqpOptimizer
from qiskit.optimization.algorithms.goemans_williamson_optimizer import GoemansWilliamsonOptimizer
from qiskit.optimization.algorithms.warm_start_qaoa_optimizer import WarmStartQAOAOptimizer, \
    MeanAggregator
from qiskit.optimization.applications.ising.max_cut import max_cut_qp


class TestWarmStartQAOAOptimizer(QiskitOptimizationTestCase):
    """Tests for the warm start QAOA optimizer."""

    def test_max_cut(self):
        """Basic test on the max cut problem."""
        graph = np.array([[0., 1., 2., 0.],
                          [1., 0., 1., 0.],
                          [2., 1., 0., 1.],
                          [0., 0., 1., 0.]])

        presolver = GoemansWilliamsonOptimizer(num_cuts=10)
        problem = max_cut_qp(graph)

        backend = BasicAer.get_backend("statevector_simulator")
        qaoa = QAOA(quantum_instance=backend, p=1)
        aggregator = MeanAggregator()
        optimizer = WarmStartQAOAOptimizer(pre_solver=presolver, relax_for_pre_solver=False,
                                           qaoa=qaoa, epsilon=0.25, num_initial_solutions=10,
                                           aggregator=aggregator)
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_almost_equal([0, 0, 1, 0], result_warm.x, 3)
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_almost_equal(4, result_warm.fval, 3)

    def test_constrained_binary(self):
        """Constrained binary optimization problem."""
        model = Model()
        v = model.binary_var(name="v")
        w = model.binary_var(name="w")
        # pylint:disable=invalid-name
        t = model.binary_var(name="t")

        model.minimize(v + w + t)
        model.add_constraint(2 * v + 10 * w + t <= 3, "cons1")
        model.add_constraint(v + w + t >= 2, "cons2")

        problem = QuadraticProgram()
        problem.from_docplex(model)

        backend = BasicAer.get_backend("statevector_simulator")
        qaoa = QAOA(quantum_instance=backend, p=1)
        aggregator = MeanAggregator()
        optimizer = WarmStartQAOAOptimizer(pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True,
                                           qaoa=qaoa, epsilon=0.25, aggregator=aggregator)
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_almost_equal([1, 0, 1], result_warm.x, 3)
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_almost_equal(2, result_warm.fval, 3)

    def test_simple_qubo(self):
        """Test on a simple QUBO problem."""
        model = Model()
        # pylint:disable=invalid-name
        u = model.binary_var(name="u")
        v = model.binary_var(name="v")

        model.minimize((u - v + 2) ** 2)
        problem = QuadraticProgram()
        problem.from_docplex(model)

        backend = BasicAer.get_backend("statevector_simulator")
        qaoa = QAOA(quantum_instance=backend, p=1)
        optimizer = WarmStartQAOAOptimizer(pre_solver=SlsqpOptimizer(), relax_for_pre_solver=True,
                                           qaoa=qaoa, epsilon=0.25)
        result_warm = optimizer.solve(problem)

        self.assertIsNotNone(result_warm)
        self.assertIsNotNone(result_warm.x)
        np.testing.assert_almost_equal([0, 1], result_warm.x, 3)
        self.assertIsNotNone(result_warm.fval)
        np.testing.assert_almost_equal(1, result_warm.fval, 3)
