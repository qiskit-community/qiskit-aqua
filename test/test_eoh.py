# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

import numpy as np
from qiskit.transpiler import PassManager
from qiskit import BasicAer
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

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend, shots=1, max_credits=10, pass_manager=PassManager())
        # self.log.debug('state_out:\n\n')

        ret = eoh.run(quantum_instance)
        self.log.debug('Evaluation result: {}'.format(ret))


if __name__ == '__main__':
    unittest.main()
