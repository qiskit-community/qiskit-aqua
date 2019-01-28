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
from qiskit.transpiler import PassManager
from qiskit.aqua import get_aer_backend
from qiskit.qobj import RunConfig
from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.algorithms import EOH


class TestEOH(QiskitAquaTestCase):
    """Evolution tests."""

    def test_eoh(self):
        SIZE = 2

        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        qubit_op = Operator(matrix=h1)

        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        evo_op = Operator(matrix=h1)

        state_in = Custom(SIZE, state='random')

        evo_time = 1
        num_time_slices = 100

        eoh = EOH(qubit_op, state_in, evo_op, 'paulis', evo_time, num_time_slices)

        backend = get_aer_backend('statevector_simulator')
        run_config = RunConfig(shots=1, max_credits=10, memory=False)
        quantum_instance = QuantumInstance(backend, run_config, pass_manager=PassManager())
        # self.log.debug('state_out:\n\n')

        ret = eoh.run(quantum_instance)
        self.log.debug('Evaluation result: {}'.format(ret))


if __name__ == '__main__':
    unittest.main()
