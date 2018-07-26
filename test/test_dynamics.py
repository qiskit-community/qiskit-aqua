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

import numpy as np

from test.common import QiskitAquaTestCase
from qiskit_aqua.operator import Operator
from qiskit_aqua import get_algorithm_instance, get_initial_state_instance


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_evolution(self):
        SIZE = 2

        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        qubitOp = Operator(matrix=h1)

        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        evoOp = Operator(matrix=h1)

        state_in = get_initial_state_instance('CUSTOM')
        state_in.init_args(SIZE, state='random')

        evo_time = 1
        num_time_slices = 100

        dynamics = get_algorithm_instance('Dynamics')
        dynamics.setup_quantum_backend(skip_transpiler=True)
        # self.log.debug('state_out:\n\n')

        dynamics.init_args(qubitOp, 'paulis', state_in, evoOp, evo_time, num_time_slices)
        ret = dynamics.run()
        self.log.debug('Evaluation result: {}'.format(ret))


if __name__ == '__main__':
    unittest.main()
