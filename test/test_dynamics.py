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

import numpy as np
import unittest
from qiskit.tools.qi.qi import state_fidelity
from qiskit import QuantumRegister
from test.common import QISKitAcquaTestCase
from qiskit_acqua.operator import Operator
from qiskit_acqua import get_algorithm_instance, get_initial_state_instance
from qiskit.wrapper import execute as q_execute


# @unittest.skipUnless(QISKitAcquaTestCase.SLOW_TEST, 'slow')
class TestEvolution(QISKitAcquaTestCase):
    """Evolution tests."""

    def test_evolution(self):
        SIZE = 2
        #SPARSITY = 0
        #X = [[0, 1], [1, 0]]
        #Y = [[0, -1j], [1j, 0]]
        Z = [[1, 0], [0, -1]]
        I = [[1, 0], [0, 1]]
        h1 = np.kron(I, Z)# + 0.5 * np.kron(Y, X)# + 0.3 * np.kron(Z, X) + 0.4 * np.kron(Z, Y)

        # np.random.seed(2)
        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        qubitOp = Operator(matrix=h1)

        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        evoOp = Operator(matrix=h1)

        state_in = get_initial_state_instance('CUSTOM')
        state_in.init_args(SIZE, state='random')

        evo_time = 1
        num_time_slices = 1

        dynamics = get_algorithm_instance('Dynamics')
        dynamics.setup_quantum_backend()
        # self.log.debug('state_out:\n\n')

        dynamics.init_args(qubitOp, 'paulis', state_in, evoOp, evo_time, num_time_slices)
        ret = dynamics.run()
        self.log.debug('Evaluation result: {}'.format(ret))


if __name__ == '__main__':
    unittest.main()

