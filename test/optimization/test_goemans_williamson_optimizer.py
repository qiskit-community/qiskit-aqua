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

""" Test Goemans-Williamson optimizer. """

from test.optimization import QiskitOptimizationTestCase

import numpy as np

from qiskit.aqua import MissingOptionalLibraryError
from qiskit.optimization.algorithms.goemans_williamson_optimizer \
    import GoemansWilliamsonOptimizer, GoemansWilliamsonOptimizationResult
from qiskit.optimization.applications.ising.max_cut import max_cut_qp


class TestGoemansWilliamson(QiskitOptimizationTestCase):
    """Test Goemans-Williamson optimizer."""
    def test_all_cuts(self):
        """Basic test of the Goemans-Williamson optimizer."""
        try:
            graph = np.array([[0., 1., 2., 0.],
                              [1., 0., 1., 0.],
                              [2., 1., 0., 1.],
                              [0., 0., 1., 0.]])

            optimizer = GoemansWilliamsonOptimizer(num_cuts=10, seed=0)

            problem = max_cut_qp(graph)
            self.assertIsNotNone(problem)

            results = optimizer.solve(problem)
            self.assertIsNotNone(results)
            self.assertIsInstance(results, GoemansWilliamsonOptimizationResult)
            self.assertIsNotNone(results.x)
            np.testing.assert_almost_equal([0, 1, 1, 0], results.x, 3)
            self.assertIsNotNone(results.fval)
            np.testing.assert_almost_equal(4, results.fval, 3)
            self.assertIsNotNone(results.explored_solutions)
            self.assertEqual(3, len(results.explored_solutions))
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
