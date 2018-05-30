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

"""
InputParser test.
"""

import unittest
from test.common import QISKitAcquaTestCase
import json

from qiskit_acqua import run_algorithm


@unittest.skipUnless(QISKitAcquaTestCase.SLOW_TEST, 'slow')
class TestAlgorithms(QISKitAcquaTestCase):
    """InputParser tests."""

    def test_algo_from_json(self):
        filepath = self._get_resource_path('algo.json')
        params = None
        with open(filepath) as json_file:
            params = json.load(json_file)

        run_algorithm(params, None, True)


if __name__ == '__main__':
    unittest.main()
