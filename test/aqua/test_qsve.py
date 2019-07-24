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

# Imports
from copy import deepcopy
import numpy as np
import unittest
# TODO: Change the line below to this line before PR: from test.aqua.common import QiskitAquaTestCase
from common import QiskitAquaTestCase
from qiskit.aqua.components.qsve import BinaryTree, QSVE
from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer


class TestQSVE(QiskitAquaTestCase):
    """Unit tests for QSVE class."""
    @staticmethod
    def final_state(circuit):
        """Returns the final state of the circuit as a numpy.ndarray."""
        # Get the unitary simulator backend
        sim = BasicAer.get_backend("unitary_simulator")

        # Execute the circuit
        job = execute(circuit, sim)

        # Get the final state
        unitary = np.array(job.result().results[0].data.unitary)
        return unitary[:, 0]

    def test_create_qsve(self):
        """Basic test for instantiating a QSVE object."""
        # Matrix to perform QSVE on
        matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)

        # Create a QSVE object
        qsve = QSVE(matrix)

        # Basic checks
        self.assertTrue(np.allclose(qsve.matrix, matrix))
        self.assertEqual(qsve.matrix_ncols, 2)
        self.assertEqual(qsve.matrix_nrows, 2)

    def test_qsve_norm(self):
        """Tests correctness for computing the norm."""
        # Matrix to perform QSVE on
        matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)

        # Create a QSVE object
        qsve = QSVE(matrix)

        # Make sure the Froebenius norm is correct
        self.assertTrue(np.isclose(qsve.matrix_norm(), np.sqrt(2)))

    def test_norm_random(self):
        """Tests correctness for computing the norm on random matrices."""
        for _ in range(100):
            # Matrix to perform QSVE on
            matrix = np.random.rand(4, 4)
            matrix += matrix.conj().T

            # Create a QSVE object
            qsve = QSVE(matrix)

            # Make sure the Froebenius norm is correct
            correct = np.linalg.norm(matrix)
            self.assertTrue(np.isclose(qsve.matrix_norm(), correct))

    def test_shift_identity(self):
        """Tests shifting the identity matrix in QSVE."""
        # Matrix for QSVE
        matrix = np.identity(2)

        # Get a QSVE object
        qsve = QSVE(matrix)

        # Shift the matrix
        qsve.shift_matrix()

        # Get the correct shifted matrix
        correct = matrix + np.linalg.norm(matrix) * np.identity(2)

        # Make sure the QSVE shifted matrix is correct
        self.assertTrue(np.allclose(qsve.matrix, correct))

    def test_shift(self):
        """Tests shifting an input matrix to QSVE."""
        # Matrix for QSVE
        matrix = np.array([[1, 2], [2, 4]], dtype=np.float64)

        # Compute the correct norm for testing the shift
        norm_correct = np.linalg.norm(matrix)

        # Get a QSVE object
        qsve = QSVE(matrix)

        # Get the BinaryTree's (one for each row of the matrix)
        tree1 = deepcopy(qsve.get_tree(0))
        tree2 = deepcopy(qsve.get_tree(1))

        # Shift the matrix
        qsve.shift_matrix()

        # Get the correct shifted matrix
        correct = matrix + norm_correct * np.identity(2)

        # Make sure the QSVE shifted matrix is correct
        self.assertTrue(np.allclose(qsve.matrix, correct))

        # Get the new BinaryTrees after shifting
        new_tree1 = qsve.get_tree(0)
        new_tree2 = qsve.get_tree(1)

        # Get the new correct tree values
        correct_new_tree1_values = np.array([tree1._values[0] + norm_correct, tree1._values[1]])
        correct_new_tree2_values = np.array([tree2._values[0], tree2._values[1] + norm_correct])

        # Make sure the BinaryTrees in the qsve object were updated correctly
        self.assertTrue(np.array_equal(new_tree1._values, correct_new_tree1_values))
        self.assertTrue(np.array_equal(new_tree2._values, correct_new_tree2_values))

    def test_row_norm_tree(self):
        """Tests creating the row norm tree for a matrix."""
        # Test matrix
        matrix = np.array([[1, 1, 0, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0],
                           [0, 1, 1, 0]], dtype=np.float64)

        # Make it Hermitian
        matrix += matrix.conj().T

        # Create the QSVE object
        qsve = QSVE(matrix)

        # Calculate the correct two-norms for each row of the matrix
        two_norms = np.array([np.linalg.norm(row, ord=2) for row in matrix])

        self.assertTrue(np.allclose(two_norms, qsve.row_norm_tree._values))
        self.assertTrue(np.allclose(two_norms ** 2, qsve.row_norm_tree.leaves))
        self.assertTrue(np.isclose(qsve.row_norm_tree.root, np.linalg.norm(matrix, "fro") ** 2))

    def test_row_norm_tree_random(self):
        """Tests correctness of the row norm tree for random matrices."""
        for _ in range(100):
            # Get a random matrix
            matrix = np.random.randn(8, 8)

            # Make it Hermitian
            matrix += matrix.conj().T

            # Create the QSVE object
            qsve = QSVE(matrix)

            # Calculate the correct two-norms for each row of the matrix
            two_norms = np.array([np.linalg.norm(row, ord=2) for row in matrix])

            self.assertTrue(np.allclose(two_norms, qsve.row_norm_tree._values))
            self.assertTrue(np.allclose(two_norms ** 2, qsve.row_norm_tree.leaves))
            self.assertTrue(np.isclose(qsve.row_norm_tree.root, np.linalg.norm(matrix, "fro") ** 2))

    def test_row_norm_tree_prep_circuit(self):
        """Tests the state preparation circuit for the row norm tree."""
        # Test matrix
        matrix = np.array([[1, 1, 0, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 1],
                           [0, 1, 0, 0]], dtype=np.float64)

        # Make it Hermitian
        matrix += matrix.conj().T

        # Create the QSVE object
        qsve = QSVE(matrix)

        # Calculate the correct two-norms for each row of the matrix.
        # This vector is the state the prep circuit should make.
        two_norms = np.array([np.linalg.norm(row, ord=2) for row in matrix]) / np.linalg.norm(matrix, "fro")

        # Get a register to prepare the row norm state in
        register = QuantumRegister(2)

        # Get the state preparation circuit
        circ = qsve.row_norm_tree.preparation_circuit(register)

        # Add a swap gate to get the amplitudes in a sensible order
        circ.swap(register[0], register[1])

        # Get the final state of the circuit
        state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(state, two_norms))


if __name__ == "__main__":
    unittest.main()
