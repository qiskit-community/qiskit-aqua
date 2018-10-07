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

from parameterized import parameterized
from scipy.optimize import rosen
import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua import get_optimizer_instance


class TestNLOptOptimizers(QiskitAquaTestCase):

    def setUp(self):
        np.random.seed(50)
        try:
            import nlopt
        except ImportError:
            self.skipTest('NLOpt dependency does not appear to be installed')
        pass

    def _optimize(self, optimizer):
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-6, 6)]*len(x0)
        res = optimizer.optimize(len(x0), rosen, initial_point=x0, variable_bounds=bounds)
        np.testing.assert_array_almost_equal(res[0], [1.0]*len(x0), decimal=2)
        return res

    # ESCH and ISRES do not do well with rosen
    @parameterized.expand([
        ['CRS'],
        ['DIRECT_L'],
        ['DIRECT_L_RAND'],
        # ['ESCH'],
        # ['ISRES']
    ])
    def test_nlopt(self, name):
        optimizer = get_optimizer_instance(name)
        optimizer.set_options(**{'max_evals': 50000})
        optimizer.init_args()
        res = self._optimize(optimizer)
        self.assertLessEqual(res[2], 50000)


if __name__ == '__main__':
    unittest.main()
