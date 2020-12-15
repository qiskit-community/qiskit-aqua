# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test EOH """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
from ddt import ddt, data

import numpy as np

from qiskit import BasicAer, QuantumCircuit
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.algorithms import EOH


@ddt
class TestEOH(QiskitAquaTestCase):
    """Evolution tests."""

    @data('initial_state', 'circuit')
    def test_eoh(self, mode):
        """ EOH test """
        size = 2
        aqua_globals.random_seed = 0

        temp = aqua_globals.random.random((2 ** size, 2 ** size))
        h_1 = temp + temp.T
        qubit_op = MatrixOperator(matrix=h_1)

        temp = aqua_globals.random.random((2 ** size, 2 ** size))
        h_1 = temp + temp.T
        evo_op = MatrixOperator(matrix=h_1)

        if mode == 'initial_state':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                state_in = Custom(size, state='random')
        else:
            random_state = aqua_globals.random.random(2 ** size)
            random_state = random_state / np.linalg.norm(random_state)
            state_in = QuantumCircuit(size)
            state_in.initialize(random_state, range(size))

        evo_time = 1
        num_time_slices = 100

        eoh = EOH(qubit_op, state_in, evo_op, evo_time=evo_time, num_time_slices=num_time_slices)

        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend, shots=1)
        # self.log.debug('state_out:\n\n')

        ret = eoh.run(quantum_instance)
        self.log.debug('Evaluation result: %s', ret)


if __name__ == '__main__':
    unittest.main()
