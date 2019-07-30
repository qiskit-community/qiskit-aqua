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

        # Get a register and circuit to prepare the row norm state in
        register = QuantumRegister(2)
        circ = QuantumCircuit(register)

        # Get the state preparation circuit
        qsve.row_norm_tree.preparation_circuit(circ, register)

        # Add a swap gate to get the amplitudes in a sensible order
        circ.swap(register[0], register[1])

        # Get the final state of the circuit
        state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(state, two_norms))

    def test_controlled_unitary(self):
        pass

    def test_controlled_reflection(self):
        pass

    def test_qft(self):
        pass

    def test_singular_values_from_theta_values_for_two_by_two_identity_matrix(self):
        """Tests the correct theta values (in the range [0, 1] are measured for the identity matrix.

        These theta values are 0.25 and 0.75, or 0.01 and 0.11 as binary decimals, respectively.

        The identity matrix A = [[1, 0], [0, 1]] has singular value 1 and Froebenius norm sqrt(2). It follows that

                                    sigma / ||A||_F = 1 / sqrt(2)

        Since                       cos(pi * theta) = sigma / ||A||_F,

        we must have                cos(pi * theta) = 1 / sqrt(2),

        which means that theta = - 0.25 or theta = 0.25. After mapping from the interval [-1/2, 1/2] to the interval
        [0, 1] via

                                    theta ----> theta           (if 0 <= theta <= 1 / 2)
                                    theta ----> theta + 1       (if -1 / 2 <= theta < 0)

        (which is what we measure in QSVE), the possible outcomes are thus 0.25 and 0.75. These correspond to binary
        decimals 0.01 and 0.10, respectively.

        This test does QSVE on the identity matrix using 2, 3, 4, 5, and 6 precision qubits for QPE.
        """
        # Define the identity matrix
        matrix = np.identity(2)

        for nprecision_bits in [2, 3, 4, 5, 6]:
            # Create the QSVE instance
            qsve = QSVE(matrix, nprecision_bits=nprecision_bits)

            # Get the circuit to perform QSVE with terminal measurements on the QPE register
            circuit = qsve.create_circuit(terminal_measurements=True)

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Make sure there are only two measured values
            self.assertEqual(len(thetas_binary), 2)

            # Convert the measured angles to floating point values
            thetas = [qsve.binary_decimal_to_float(binary_decimal) for binary_decimal in thetas_binary]
            thetas = [qsve.convert_measured(theta) for theta in thetas]

            # Make sure the theta values are correct
            self.assertEqual(len(thetas), 2)
            self.assertIn(0.25, thetas)
            self.assertIn(-0.25, thetas)

    def test_singular_values_two_by_two_pi_over_eight(self):
        """Tests computing the singular values of the matrix

                    A = [[cos(pi / 8), 0],
                         [0, sin(pi / 8)]]

        The QSVE algorithm should be able to compute the singular values exactly with three qubits (or more).
        """
        # Define the matrix
        matrix = np.array([[np.cos(np.pi / 8), 0], [0, np.sin(np.pi / 8)]])

        # Do the classical SVD. (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            qsve = QSVE(matrix, nprecision_bits=nprecision_bits)
            circuit = qsve.create_circuit(terminal_measurements=True)

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits)) for bits in thetas_binary]

            # Convert from theta values to singular values
            qsigmas = [np.cos(np.pi * theta) for theta in computed if theta > 0]

            # Sort the sigma values for comparison
            sigmas = list(sorted(sigmas))
            qsigmas = list(sorted(qsigmas))

            # Make sure the quantum solution is close to the classical solution
            self.assertTrue(np.allclose(sigmas, qsigmas))

    def test_singular_values_two_by_two_three_pi_over_eight(self):
        """Tests computing the singular values of the matrix

                    A = [[cos(3 * pi / 8), 0],
                         [0, sin(3 * pi / 8)]]

        The QSVE algorithm should be able to compute the singular values exactly with three qubits (or more).
        """
        # Define the matrix
        matrix = np.array([[np.cos(3 * np.pi / 8), 0], [0, np.sin(3 * np.pi / 8)]])

        # Do the classical SVD. (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            qsve = QSVE(matrix, nprecision_bits=nprecision_bits)
            circuit = qsve.create_circuit(terminal_measurements=True)

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits)) for bits in thetas_binary]

            # Convert from theta values to singular values
            qsigmas = [np.cos(np.pi * theta) for theta in computed if theta > 0]

            # Sort the sigma values for comparison
            sigmas = list(sorted(sigmas))
            qsigmas = list(sorted(qsigmas))

            # Make sure the quantum solution is close to the classical solution
            self.assertTrue(np.allclose(sigmas, qsigmas))

    def test_row_isometry_two_by_two(self):
        """Tests that the row isometry is indeed an isometry for random two by two matrices."""
        iden = np.identity(2)
        for _ in range(100):
            # Get a Hermitian matrix
            matrix = np.random.randn(2, 2)
            matrix += matrix.conj().T

            # Create the QSVE object and get the row isometry
            qsve = QSVE(matrix)
            umat = qsve.row_isometry()
            vmat = qsve.norm_isometry()

            # Make sure U^dagger U = I
            self.assertTrue(np.allclose(umat.conj().T @ umat, iden))
            self.assertTrue(np.allclose(vmat.conj().T @ vmat, iden))

            # Make sure U^dagger V = V^dagger U = A / ||A||_F
            self.assertTrue(np.allclose(umat.conj().T @ vmat, matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat, matrix / np.linalg.norm(matrix, ord="fro")))

    def test_isometries_four_by_four(self):
        """Tests that the row (norm) isometry is indeed an isometry for random four by four matrices."""
        iden = np.identity(4)
        for _ in range(100):
            # Get a Hermitian matrix
            matrix = np.random.randn(4, 4)
            matrix += matrix.conj().T

            # Get a QSVE object and compute the row ismoetry
            qsve = QSVE(matrix)
            umat = qsve.row_isometry()
            vmat = qsve.norm_isometry()

            # Make sure U^dagger U = I = V^dagger V
            self.assertTrue(np.allclose(umat.conj().T @ umat, iden))
            self.assertTrue(np.allclose(vmat.conj().T @ vmat, iden))

            # Make sure U^dagger V = V^dagger U = A / ||A||_F
            self.assertTrue(np.allclose(umat.conj().T @ vmat, matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat, matrix / np.linalg.norm(matrix, ord="fro")))

            # Make sure rank(U U^dagger) = 4
            self.assertEqual(np.linalg.matrix_rank(umat @ umat.conj().T), len(matrix))

    def test_isometries_large_matrix(self):
        """Tests that the row isometry is indeed an isometry for random 16 x 16 matrices."""
        iden = np.identity(16)
        for _ in range(100):
            # Get a random Hermitian matrix
            matrix = np.random.randn(16, 16)
            matrix += matrix.conj().T

            # Get the row isometry
            qsve = QSVE(matrix)
            umat = qsve.row_isometry()
            vmat = qsve.norm_isometry()

            # Make sure U^dagger U = I
            self.assertTrue(np.allclose(umat.conj().T @ umat, iden))
            self.assertTrue(np.allclose(vmat.conj().T @ vmat, iden))

            # Make sure U^dagger V = V^dagger U = A / ||A||_F
            self.assertTrue(np.allclose(umat.conj().T @ vmat, matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat, matrix / np.linalg.norm(matrix, ord="fro")))

            # Make sure rank(U U^dagger) = 256
            self.assertEqual(np.linalg.matrix_rank(vmat @ vmat.conj().T), len(matrix))

    def test_unitary(self):
        """Tests for the unitary used for QPE."""
        for dim in [2, 4]:
            for _ in range(50):
                matrix = np.random.randn(dim, dim)
                matrix += matrix.conj().T

                qsve = QSVE(matrix)
                unitary = qsve.unitary()

                self.assertTrue(np.allclose(unitary.conj().T @ unitary, np.identity(dim**2)))


if __name__ == "__main__":
    unittest.main()

    matrix = np.identity(2)

    qsve = QSVE(matrix, nprecision_bits=2)

    circuit = qsve.create_circuit()

    print(circuit)

    # # Define the matrix
    # matrix = np.array([[np.cos(2 * np.pi / 8), 0, 0, 0],
    #                    [0, np.sin(2 * np.pi / 8), 0, 0],
    #                    [0, 0, np.cos(2 * np.pi / 8), 0],
    #                    [0, 0, 0, np.sin(2 * np.pi / 8)]]) / np.sqrt(2)
    #
    # # Do the classical SVD. (Note: We could just access the singular values from the diagonal matrix elements.)
    # _, sigmas, _ = np.linalg.svd(matrix)
    #
    # qsve = QSVE(matrix, nprecision_bits=5)
    # circuit = qsve.create_circuit(terminal_measurements=True)
    #
    # # Run the quantum circuit for QSVE
    # sim = BasicAer.get_backend("qasm_simulator")
    # job = execute(circuit, sim, shots=50000)
    #
    # # Get the output bit strings from QSVE
    # res = job.result()
    # counts = res.get_counts()
    # thetas_binary = np.array(list(counts.keys()))
    #
    # # print("Sampled thetas:", thetas_binary)
    # #
    # # # Convert from the binary strings to theta values
    # # computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits)) for bits in thetas_binary]
    # #
    # # # Convert from theta values to singular values
    # # qsigmas = [np.cos(np.pi * theta) for theta in computed]
    # #
    # # # Sort the sigma values for comparison
    # # sigmas = list(sorted(sigmas))
    # # qsigmas = list(sorted(qsigmas))
    # #
    # # print("sigmas:", sigmas)
    # # print("qsigmas", qsigmas)
    #
    # # Get the top measured bit strings
    # import operator
    # sort = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    # print()
    # print(sort)
    #
    # top = [x[0] for x in sort[:len(sort) // 2]]
    #
    # print("\nTop sampled bit strings from QSVE:")
    # print(top)
    #
    # # Convert from the binary strings to theta values
    # computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits)) for bits in top]
    #
    # print("\nTop quantumly found theta values")
    # print(computed)
    #
    # # Convert from theta values to singular values
    # print("\nTop quantumly found singular values")
    # qsigmas = [np.cos(np.pi * theta) for theta in computed]
    # print(qsigmas)
    #
    # print("Acutal sigmas")
    # print(sigmas / np.linalg.norm(matrix, "fro"))
    #
    # print("\nMaximum theoretic error:", qsve.max_error())
