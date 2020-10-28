# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Hartree Fock initial state circuit."""

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np

from qiskit import QuantumCircuit
from qiskit.chemistry.circuit.library import HartreeFock
from qiskit.chemistry.circuit.library.initial_states.hartree_fock import hartree_fock_bitstring


class TestHartreeFock(QiskitChemistryTestCase):
    """ Initial State HartreeFock tests """

    def test_bitstring(self):
        """Simple test for the bitstring function."""
        bitstr = hartree_fock_bitstring(4, 2, 'parity', True)
        self.assertTrue(all(bitstr[::-1] == np.array([True, False])))  # big endian notation

    def test_bitstring_invalid_input(self):
        """Test passing invalid input raises."""

        with self.subTest('too many particles'):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(4, 4, 'jordan-wigner')

        with self.subTest('too few orbitals'):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(-1, 4, 'parity', True)

        with self.subTest('invalid qubit mapping'):
            with self.assertRaises(ValueError):
                _ = hartree_fock_bitstring(4, 2, 'parit', True)

    def test_qubits_4_jw_h2(self):
        """ qubits 4 jw h2 test """
        state = HartreeFock(4, (1, 1), 'jordan_wigner', False)
        ref = QuantumCircuit(4)
        ref.x([0, 2])
        self.assertEqual(state, ref)

    def test_qubits_4_py_h2(self):
        """ qubits 4 py h2 test """
        state = HartreeFock(4, (1, 1), 'parity', False)
        ref = QuantumCircuit(4)
        ref.x([0, 1])
        self.assertEqual(state, ref)

    def test_qubits_4_bk_h2(self):
        """ qubits 4 bk h2 test """
        state = HartreeFock(4, (1, 1), 'bravyi_kitaev', False)
        ref = QuantumCircuit(4)
        ref.x([0, 1, 2])
        self.assertEqual(state, ref)

    def test_qubits_2_py_h2(self):
        """ qubits 2 py h2 test """
        state = HartreeFock(4, 2, 'parity', True)
        ref = QuantumCircuit(2)
        ref.x(0)
        self.assertEqual(state, ref)

    def test_qubits_6_py_lih(self):
        """ qubits 6 py lih test """
        state = HartreeFock(10, (1, 1), 'parity', True, [1, 2])
        ref = QuantumCircuit(6)
        ref.x([0, 1])
        self.assertEqual(state, ref)


if __name__ == '__main__':
    unittest.main()
