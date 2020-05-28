# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Fixed Value Comparator """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
from ddt import ddt, idata, unpack
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute
from qiskit.aqua.circuits import FixedValueComparator as Comparator


@ddt
class TestFixedValueComparator(QiskitAquaTestCase):
    """ Text Fixed Value Comparator """

    def setUp(self):
        super().setUp()
        # ignore deprecation warnings from the change of the circuit factory to circuit library
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", category=DeprecationWarning)

    @idata([
        # n, value, geq
        [1, 0, True],
        [1, 1, True],
        [2, -1, True],
        [2, 0, True],
        [2, 1, True],
        [2, 2, True],
        [2, 3, True],
        [2, 4, True],
        [3, 5, True],
        [4, 6, False]
    ])
    @unpack
    def test_fixed_value_comparator(self, num_state_qubits, value, geq):
        """ fixed value comparator test """
        # initialize weighted sum operator factory
        comp = Comparator(num_state_qubits, value, geq)

        # initialize circuit
        q = QuantumRegister(num_state_qubits + 1)
        if comp.required_ancillas() > 0:
            q_a = QuantumRegister(comp.required_ancillas())
            qc = QuantumCircuit(q, q_a)
        else:
            q_a = None
            qc = QuantumCircuit(q)

        # set equal superposition state
        qc.h(q[:num_state_qubits])

        # build circuit
        comp.build(qc, q, q_a)

        # run simulation
        job = execute(qc, BasicAer.get_backend('statevector_simulator'), shots=1)

        for i, s_a in enumerate(job.result().get_statevector()):

            prob = np.abs(s_a)**2
            if prob > 1e-6:
                # equal superposition
                self.assertEqual(True, np.isclose(1.0, prob * 2.0 ** num_state_qubits))
                b_value = '{0:b}'.format(i).rjust(qc.width(), '0')
                x = int(b_value[(-num_state_qubits):], 2)
                comp_result = int(b_value[-num_state_qubits - 1], 2)
                if geq:
                    self.assertEqual(x >= value, comp_result == 1)
                else:
                    self.assertEqual(x < value, comp_result == 1)


if __name__ == '__main__':
    unittest.main()
