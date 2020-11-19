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

"""Test the VSCF initial state."""

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit.chemistry.circuit.library import VSCF
from qiskit.chemistry.circuit.library.initial_states.vscf import vscf_bitstring


class TestVSCF(QiskitChemistryTestCase):
    """ Initial State vscf tests """

    def test_bitstring(self):
        """Test the vscf_bitstring method."""
        bitstr = vscf_bitstring([2, 2])
        self.assertTrue(all(bitstr[::-1] == np.array([True, False, True, False])))  # big endian

    def test_qubits_4(self):
        """Test 2 modes 2 modals."""
        basis = [2, 2]
        vscf = VSCF(basis)
        ref = QuantumCircuit(4)
        ref.x([0, 2])

        self.assertEqual(ref, vscf)

    def test_qubits_5(self):
        """Test 2 modes 2 modals for the first mode and 3 modals for the second."""
        basis = [2, 3]
        vscf = VSCF(basis)
        ref = QuantumCircuit(5)
        ref.x([0, 2])

        self.assertEqual(ref, vscf)


if __name__ == '__main__':
    unittest.main()
