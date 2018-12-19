# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest

from scipy.optimize import rosen
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua.components.optimizers import (CG, COBYLA, L_BFGS_B, NELDER_MEAD,
                                               POWELL, SLSQP, SPSA, TNC)


class TestOptimizers(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(50)
        pass

    def _optimize(self, optimizer):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res = optimizer.optimize(len(x0), rosen, initial_point=x0)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x0), decimal=2)
        return res

    def test_cg(self):
        optimizer = CG(tol=1e-06)
        optimizer.set_options(**{'maxiter': 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_cobyla(self):
        optimizer = COBYLA(tol=1e-06)
        optimizer.set_options(**{'maxiter': 100000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_l_bfgs_b(self):
        optimizer = L_BFGS_B()
        optimizer.set_options(**{'maxfun': 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_nelder_mead(self):
        optimizer = NELDER_MEAD(tol=1e-06)
        optimizer.set_options(**{'maxfev': 10000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_powell(self):
        optimizer = POWELL(tol=1e-06)
        optimizer.set_options(**{'maxfev': 10000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    def test_slsqp(self):
        optimizer = SLSQP(tol=1e-06)
        optimizer.set_options(**{'maxiter': 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)

    @unittest.skip("Skipping SPSA as it does not do well on non-convex rozen")
    def test_spsa(self):
        optimizer = SPSA(max_trials=10000)
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 100000)

    def test_tnc(self):
        optimizer = TNC(tol=1e-06)
        optimizer.set_options(**{'maxiter': 1000})
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 10000)


if __name__ == '__main__':
    unittest.main()
