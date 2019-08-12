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
from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua.components.qsve import QSVE
from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer


class TestQSVE(QiskitAquaTestCase):
    """Unit tests for QSVE class."""
    @staticmethod
    def final_state(circuit):
        """Returns the final state of the circuit as a numpy.ndarray."""
        return TestQSVE.unitary_of(circuit)[:, 0]

    @staticmethod
    def unitary_of(circuit):
        """Returns the unitary of a circuit.

        Args:
            circuit : qiskit.QuantumCircuit
                Circuit to get the unitary of.

        Returns: numpy.ndarray
            Unitary matrix of the circuit
        """
        # Get the unitary simulator backend
        sim = BasicAer.get_backend("unitary_simulator")

        # Execute the circuit
        job = execute(circuit, sim)

        # Return the unitary
        return np.array(job.result().results[0].data.unitary)

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

    def test_singular_values_from_theta_values_for_two_by_two_identity_matrix(self):
        """Tests the correct theta values (in the range [0, 1] are measured for the identity matrix.

        These theta values are 0.25 and 0.75, or 0.01 and 0.11 as binary decimals, respectively.

        The identity matrix A = [[1, 0], [0, 1]] has singular value 1 and Froebenius norm sqrt(2).
        It follows that

                                    sigma / ||A||_F = 1 / sqrt(2)

        Since                       cos(pi * theta) = sigma / ||A||_F,

        we must have                cos(pi * theta) = 1 / sqrt(2),

        which means that theta = - 0.25 or theta = 0.25.
        After mapping from the interval [-1/2, 1/2] to the interval [0, 1] via

                                    theta ----> theta           (if 0 <= theta <= 1 / 2)
                                    theta ----> theta + 1       (if -1 / 2 <= theta < 0)

        (which is what we measure in QSVE), the possible outcomes are thus 0.25 and 0.75.
        These correspond to binary decimals 0.01 and 0.10, respectively.

        This test does QSVE on the identity matrix using 2, 3, 4, 5, and 6 precision qubits for QPE.
        """
        # Define the identity matrix
        matrix = np.identity(2)

        # Create the QSVE instance
        qsve = QSVE(matrix)

        for nprecision_bits in [2, 3, 4, 5, 6]:
            # Get the circuit to perform QSVE with terminal measurements on the QPE register
            circuit = qsve.create_circuit(nprecision_bits=nprecision_bits, terminal_measurements=True)

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
            thetas = [qsve.binary_decimal_to_float(binary_decimal, big_endian=False)
                      for binary_decimal in thetas_binary]
            thetas = [qsve.convert_measured(theta) for theta in thetas]

            # Make sure the theta values are correct
            self.assertEqual(len(thetas), 2)
            self.assertIn(0.25, thetas)
            self.assertIn(-0.25, thetas)

    def test_singular_values_two_by_two_pi_over_eight(self):
        """Tests computing the singular values of the matrix

                    A = [[cos(pi / 8), 0],
                         [0, sin(pi / 8)]]

        The QSVE algorithm should be able to compute the singular values
        exactly with three qubits (or more).
        """
        # Define the matrix
        matrix = np.array([[np.cos(np.pi / 8), 0], [0, np.sin(np.pi / 8)]])

        # Do the classical SVD. (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        qsve = QSVE(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            circuit = qsve.create_circuit(
                nprecision_bits=nprecision_bits,
                init_state_row_and_col=[1, 1, 1, 1],
                terminal_measurements=True
            )
            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits))
                        for bits in thetas_binary]

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

        # Do the classical SVD.
        # (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            qsve = QSVE(matrix)
            circuit = qsve.create_circuit(
                nprecision_bits=nprecision_bits,
                init_state_row_and_col=[1, 1, 1, 1],
                terminal_measurements=True
            )

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits))
                        for bits in thetas_binary]

            # Convert from theta values to singular values
            qsigmas = [np.cos(np.pi * theta) for theta in computed if theta > 0]

            # Sort the sigma values for comparison
            sigmas = list(sorted(sigmas))
            qsigmas = list(sorted(qsigmas))

            # Make sure the quantum solution is close to the classical solution
            self.assertTrue(np.allclose(sigmas, qsigmas))

    def test_singular_values_two_by_two_three_pi_over_eight_singular_vector(self):
        """Tests computing the singular values of the matrix

                    A = [[cos(3 * pi / 8), 0],
                         [0, sin(3 * pi / 8)]]

        with an input singular vector.
        Checks that only one singular value is present in the measurement outcome.
        """
        # Define the matrix
        matrix = np.array([[np.cos(3 * np.pi / 8), 0], [0, np.sin(3 * np.pi / 8)]])

        # Do the classical SVD.
        # (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            qsve = QSVE(matrix)
            circuit = qsve.create_circuit(
                nprecision_bits=nprecision_bits,
                init_state_row_and_col=[1, 0, 0, 0],
                terminal_measurements=True
            )

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits))
                        for bits in thetas_binary]

            # Convert from theta values to singular values
            qsigmas = [np.cos(np.pi * theta) for theta in computed if theta > 0]

            # Sort the sigma values for comparison
            sigmas = list(sorted(sigmas))
            qsigmas = list(sorted(qsigmas))

            # Make sure the quantum solution is close to the classical solution
            self.assertTrue(np.allclose(sigmas[0], qsigmas))

    def test_singular_values_two_by_two_three_pi_over_eight_singular_vector2(self):
        """Tests computing the singular values of the matrix

                    A = [[cos(3 * pi / 8), 0],
                         [0, sin(3 * pi / 8)]]

        with an input singular vector.
        Checks that only one singular value is present in the measurement outcome.
        """
        # Define the matrix
        matrix = np.array([[np.cos(3 * np.pi / 8), 0], [0, np.sin(3 * np.pi / 8)]])

        # Do the classical SVD.
        # (Note: We could just access the singular values from the diagonal matrix elements.)
        _, sigmas, _ = np.linalg.svd(matrix)

        # Get the quantum circuit for QSVE
        for nprecision_bits in range(3, 7):
            qsve = QSVE(matrix)
            circuit = qsve.create_circuit(
                nprecision_bits=nprecision_bits,
                init_state_row_and_col=[0, 1, 0, 0],
                terminal_measurements=True
            )

            # Run the quantum circuit for QSVE
            sim = BasicAer.get_backend("qasm_simulator")
            job = execute(circuit, sim, shots=10000)

            # Get the output bit strings from QSVE
            res = job.result()
            counts = res.get_counts()
            thetas_binary = np.array(list(counts.keys()))

            # Convert from the binary strings to theta values
            computed = [qsve.convert_measured(qsve.binary_decimal_to_float(bits))
                        for bits in thetas_binary]

            # Convert from theta values to singular values
            qsigmas = [np.cos(np.pi * theta) for theta in computed if theta > 0]

            # Sort the sigma values for comparison
            sigmas = list(sorted(sigmas))
            qsigmas = list(sorted(qsigmas))

            # Make sure the quantum solution is close to the classical solution
            self.assertTrue(np.allclose(sigmas[1], qsigmas))

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
            self.assertTrue(np.allclose(umat.conj().T @ vmat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))

    def test_isometries_four_by_four(self):
        """Tests that the row (norm) isometry is indeed an isometry
         for random four by four matrices."""
        iden = np.identity(4)
        for _ in range(100):
            # Get a Hermitian matrix
            matrix = np.random.randn(4, 4)
            matrix += matrix.conj().T

            # Get a QSVE object and compute the row isometry
            qsve = QSVE(matrix)
            umat = qsve.row_isometry()
            vmat = qsve.norm_isometry()

            # Make sure U^dagger U = I = V^dagger V
            self.assertTrue(np.allclose(umat.conj().T @ umat, iden))
            self.assertTrue(np.allclose(vmat.conj().T @ vmat, iden))

            # Make sure U^dagger V = V^dagger U = A / ||A||_F
            self.assertTrue(np.allclose(umat.conj().T @ vmat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))

            # Make sure rank(U U^dagger) = 4
            self.assertEqual(np.linalg.matrix_rank(umat @ umat.conj().T), len(matrix))

    def test_isometries_large_matrix(self):
        """Tests that the row isometry is indeed an isometry
        for random 16 x 16 matrices."""
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
            self.assertTrue(np.allclose(umat.conj().T @ vmat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))
            self.assertTrue(np.allclose(vmat.conj().T @ umat,
                                        matrix / np.linalg.norm(matrix, ord="fro")))

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

    def test_unitary_conjugate_evals(self):
        """Tests that for each eigenvalue lambda of the unitary, lambda* is also an eigenvalue, as required."""
        for _ in range(100):
            matrix = np.random.randn(2, 2)
            matrix += matrix.conj().T
            qsve = QSVE(matrix)

            umat = qsve.unitary()

            evals, _ = np.linalg.eig(umat)

            for eval in evals:
                self.assertIn(np.conjugate(eval), evals)

    def test_unitary_evals_matrix_singular_values_identity_2by2(self):
        """Tests that the eigenvalues of the unitary are the matrix singular values."""
        # Dimension of system
        dim = 2

        # Define the matrix and QSVE object
        matrix = np.identity(dim)
        qsve = QSVE(matrix)

        # Get the (normalized) singular values of the matrix
        sigmas = qsve.singular_values_classical(normalized=True)

        # Get the eigenvalues of the QSVE unitary
        evals, _ = np.linalg.eig(qsve.unitary())

        # Make sure there are the correct number of eigenvalues
        self.assertEqual(len(evals), dim**2)

        qsigmas = []

        for eval in evals:
            qsigma = qsve.unitary_eval_to_singular_value(eval)
            if qsigma not in qsigmas:
                qsigmas.append(qsigma)

        self.assertTrue(np.allclose(qsigmas, sigmas))

    def test_unitary_evals_to_matrix_singular_vals(self):
        """Tests QSVE.unitary() by ensuring the eigenvalues of the unitary relate to the singular values of the
        input matrix in the expected way.
        """
        for _ in range(100):
            matrix = np.random.randn(2, 2)
            matrix += matrix.conj().T
            qsve = QSVE(matrix)

            sigmas = qsve.singular_values_classical(normalized=True)

            umat = qsve.unitary()
            evals, _ = np.linalg.eig(umat)

            qsigmas = []

            for eval in evals:
                qsigma = qsve.unitary_eval_to_singular_value(eval)
                qsigma = round(qsigma, 4)
                if qsigma not in qsigmas:
                    qsigmas.append(qsigma)

            self.assertTrue(np.allclose(sorted(qsigmas), sorted(sigmas), atol=1e-3))

    def test_prepare_singular_vector(self):
        """Tests preparing a singular vector in two registers."""
        matrix = np.identity(2)

        qsve = QSVE(matrix)

        # Only use the real eigenvectors
        evecs = qsve.unitary_evecs()

        for vec in [evecs[0], evecs[1]]:
            row = QuantumRegister(1)
            col = QuantumRegister(1)
            circ = QuantumCircuit(row, col)

            qsve._prepare_singular_vector(np.real(vec), circ, row, col)
            circ.swap(row[0], col[0])

            state = self.final_state(circ)

            self.assertTrue(np.allclose(state, np.real(vec)))

    def test_controlled_reflection(self):
        """Basic test for controlled reflection circuit."""
        regA = QuantumRegister(1)
        regB = QuantumRegister(2)
        circ = QuantumCircuit(regA, regB)

        circ.x(regA)

        init_state = np.real(self.final_state(circ))

        QSVE._controlled_reflection_circuit(circ, regA[0], regB)

        final_state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(init_state, -final_state))

    def test_controlled_reflection2(self):
        """Basic test for controlled reflection circuit."""
        regA = QuantumRegister(1)
        regB = QuantumRegister(2)
        circ = QuantumCircuit(regA, regB)

        init_state = np.real(self.final_state(circ))

        QSVE._controlled_reflection_circuit(circ, regA, regB)

        final_state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(init_state, final_state))

    def test_controlled_reflection3(self):
        """Basic test for controlled reflection circuit."""
        regA = QuantumRegister(2)
        regB = QuantumRegister(5)
        circ = QuantumCircuit(regA, regB)

        circ.x(regA[0])
        circ.x(regB[0])

        init_state = np.real(self.final_state(circ))

        QSVE._controlled_reflection_circuit(circ, regA[0], regB)

        final_state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(init_state, final_state))

    def test_controlled_reflection4(self):
        """Basic test for controlled reflection circuit."""
        regA = QuantumRegister(2)
        regB = QuantumRegister(5)
        circ = QuantumCircuit(regA, regB)

        circ.x(regA[0])

        init_state = np.real(self.final_state(circ))

        QSVE._controlled_reflection_circuit(circ, regA[0], regB)

        final_state = np.real(self.final_state(circ))

        self.assertTrue(np.allclose(init_state, -final_state))

    def test_controlled_unitary(self):
        """Tests that the controlled unitary in a circuit agrees with the (classical) matrix representation."""
        pass

    def test_qft_dagger(self):
        """Visual test for the inverse QFT circuit."""
        n = 5
        qsve = QSVE(np.identity(4))
        qreg = QuantumRegister(n)
        circ = QuantumCircuit(qreg)
        qsve._iqft(circ, qreg)

        print(circ)

    def test_singular_values_identity4(self):
        """Tests QSVE gets close to the correct singular values for the 4 x 4 identity matrix."""
        qsve = QSVE(np.identity(4))

        sigma = max(qsve.singular_values_classical())

        for n in range(3, 8):

            qsigma = qsve.top_singular_values(
                nprecision_bits=n,
                init_state_row_and_col=None,
                shots=50000,
                ntop=1
            )

            self.assertTrue(abs(sigma - qsigma[0]) < qsve.max_error(n))

    def test_singular_values_identity8(self):
        """Tests QSVE gets close to the correct singular values for the 8 x 8 identity matrix."""
        qsve = QSVE(np.identity(8))

        sigma = max(qsve.singular_values_classical())

        for n in range(3, 7):
            qsigma = qsve.top_singular_values(
                nprecision_bits=n,
                init_state_row_and_col=None,
                shots=50000,
                ntop=3
            )

            self.assertTrue(abs(sigma - qsigma[0]) < qsve.max_error(n))

    def test_singular_values_random2x2(self):
        """Tests computing the singular values for random 2 x 2 matrices."""

        for _ in range(10):
            matrix = np.random.randn(2, 2)
            qsve = QSVE(matrix)

            n = 6
            qsigmas = qsve.top_singular_values(
                nprecision_bits=n,
                init_state_row_and_col=None,
                shots=50000,
                ntop=4
            )

            self.assertTrue(qsve.has_value_close_to_singular_values(qsigmas, qsve.max_error(n)))

    # Note: This test is commented out because it takes a while (~40 minutes) to run
    # def test_singular_values_random4x4(self):
    #     """Tests computing the singular values for random 4 x 4 matrices."""
    #     for _ in range(10):
    #         matrix = np.random.randn(4, 4)
    #         qsve = QSVE(matrix)
    #
    #         n = 6
    #         qsigmas = qsve.top_singular_values(
    #             nprecision_bits=n,
    #             init_state_row_and_col=None,
    #             shots=50000,
    #             ntop=4
    #         )
    #
    #         self.assertTrue(qsve.has_value_close_to_singular_values(qsigmas, qsve.max_error(n)))

    # Note: This test takes a while (>= an hour) to run
    # def test_singular_values_random8x8(self):
    #     """Tests computing the singular values for random 8 x 8 matrices."""
    #     for _ in range(10):
    #         matrix = np.random.randn(8, 8)
    #         matrix += matrix.conj().T
    #         qsve = QSVE(matrix)
    #
    #         print("Matrix:")
    #         print(matrix)
    #
    #         sigmas = qsve.singular_values_classical()
    #
    #         print("Sigmas:", sigmas)
    #
    #         n = 3
    #
    #         qsigmas = qsve.top_singular_values(
    #             nprecision_bits=n,
    #             singular_vector=None,
    #             shots=50000,
    #             ntop=4
    #         )
    #
    #         print("QSigmas:", qsigmas)
    #
    #         print("Max theory error:", qsve.max_error(n))
    #
    #         self.assertTrue(qsve.has_value_close_to_singular_values(qsigmas, qsve.max_error(n)))
    #         print("Success!\n\n")

    def test_possible_measured_singular_values(self):
        """Tests correctness for possible measured singular values."""
        for nbits in range(1, 10):
            vals = QSVE.possible_estimated_singular_values(nbits)
            self.assertEqual(len(vals), 2**(nbits - 1) + 1)

    def test_binary_decimal_to_float(self):
        """Tests conversion of a string binary decimal to a float."""
        for n in range(10):
            binary_decimal = "1" + "0" * n
            self.assertEqual(QSVE.binary_decimal_to_float(binary_decimal, big_endian=True), 0.5)
            self.assertEqual(QSVE.binary_decimal_to_float(binary_decimal, big_endian=False), 2**(-n - 1))

    def test_non_square(self):
        """Tests QSVE on a simple non-square matrix."""
        matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        qsve = QSVE(matrix)
        qsigmas = qsve.top_singular_values(nprecision_bits=2, ntop=2)
        self.assertTrue(qsve.has_value_close_to_singular_values(qsigmas, qsve.max_error(2)))

    def test_non_square_random(self):
        """Tests QSVE on a non-square random matrix."""
        for _ in range(10):
            matrix = np.random.randn(2, 4)
            qsve = QSVE(matrix)
            qsigmas = qsve.top_singular_values(nprecision_bits=4, ntop=-1)
            self.assertTrue(qsve.has_value_close_to_singular_values(qsigmas, qsve.max_error(4)))

    def test_binary_decimal_to_float_conversion(self):
        """Tests converting binary decimals (e.g., 0.10 = 0.5 or 0.01 = 0.25) to floats, and vice versa."""
        for num in np.linspace(0, 0.99, 25):
            self.assertAlmostEqual(
                QSVE.binary_decimal_to_float(QSVE.to_binary_decimal(num, nbits=30), big_endian=True), num
            )


if __name__ == "__main__":
    unittest.main()
