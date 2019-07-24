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

"""Module for Quantum Singular Value Estimation (QSVE).

QSVE is used as a subroutine in quantum recommendation systems [1],
where it was first introduced, and in linear systems solvers [2].

QSVE performs quantum phase estimation with the unitary W defined as

W =



References:

[1] Kerenedis and Prakash, Quantum Recommendation Systems.

[2] , , and Prakash, Quantum algorithm for dense linear systems of equations.


"""

# Imports
from copy import deepcopy
import numpy as np

from qiskit.aqua.circuits.gates.multi_control_toffoli_gate import mct
from qiskit.aqua.components.qsve import BinaryTree
from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer, transpile


class QSVE:
    """Quantum Singular Value Estimation (QSVE) class."""
    def __init__(self, matrix, nprecision_bits=3):
        """Initializes a QSVE object.

        Args:
            matrix : numpy.ndarray
                The matrix to perform singular value estimation on.

            nprecision_bits : int
                The number of qubits to use in phase estimation.
                Equivalently, the number of bits of precision to read out singular values.
        """
        # Make sure the matrix is of the correct type
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input matrix must be of type numpy.ndarray.")

        # Get the number of rows and columns in the matrix
        nrows, ncols = matrix.shape

        # Make sure the matrix is square
        # TODO: Pad matrix to automatically make it square
        if nrows != ncols:
            raise ValueError("Input matrix must be square.")

        # Make sure the number of columns is supported
        if ncols & (ncols - 1) != 0:
            raise ValueError("Number of columns in matrix must be a power of two.")

        # Make sure the matrix is Hermitian
        if not np.allclose(matrix.conj().T, matrix):
            raise ValueError("Input matrix must be Hermitian.")

        # Store these as attributes
        self._matrix_nrows = nrows
        self._matrix_ncols = ncols

        # Store the number of qubits needed for the matrix rows and cols
        self._num_qubits_for_row = int(np.log2(ncols))
        self._num_qubits_for_col = int(np.log2(nrows))
        self._num_qubits_for_qpe = int(nprecision_bits)

        # Get the number of qubits needed for the circuit
        nqubits = int(np.log2(nrows * ncols) + nprecision_bits)

        # Store a copy of the matrix
        self._matrix = deepcopy(matrix)

        # Get BinaryTree objects for each row of the matrix
        self._trees = []
        for row in matrix:
            self._trees.append(BinaryTree(row))

        # Get the "row norm tree"
        self._row_norm_tree = self._make_row_norm_tree()

        # Flag to indicate whether the matrix has been shifted or not
        self._shifted = False

    @property
    def matrix(self):
        """The matrix to perform singular value estimation on."""
        return self._matrix

    @property
    def matrix_nrows(self):
        """The number of rows in the matrix."""
        return self._matrix_nrows

    @property
    def matrix_ncols(self):
        """The number of columns in the matrix."""
        return self._matrix_ncols

    def get_tree(self, index):
        """Returns the BinaryTree representing a matrix row."""
        return self._trees[index]

    @property
    def row_norm_tree(self):
        return self._make_row_norm_tree()

    def matrix_norm(self):
        """Returns the Froebenius norm of the matrix."""
        # Compute the value using the BinaryTree's storing matrix rows.
        # With this data structure, the Froebenius norm is the sum of all roots
        value = 0.0
        for tree in self._trees:
            value += tree.root
        return np.sqrt(value)

    def _make_row_norm_tree(self):
        """Creates the BinaryTree of row norms.

        Each row norm is the square root of the root of each matrix tree.

        Example:

            Let the matrix of the QSVE be:

                    A = [[0, 1],
                         [1, 0]]

            The tree for row_0(A) = [0, 1] is:      | The tree for row_1(A) = [1, 0] is:
                                                    |
                        1.00                        |           1.00
                0.00            1.00                |   1.00            0.00

            This method first creates the vector of row norms [ ||row_0(A)||, ||row_1(A)|| ] = [1, 1], and uses this to
            create a BinaryTree, called the "row norm tree." In this case, the row norm tree would be:

                                                    2.00
                                            1.00            1.00

        Returns:
            The "row norm tree" described above.

        Return type:
            BinaryTree
        """
        # Initialize an empty list for the vector of row norms
        row_norm_vector = []

        # Append all the row norms
        for tree in self._trees:
            row_norm_vector.append(np.sqrt(tree.root))

        # Return the BinaryTree made from the row norm vector
        return BinaryTree(row_norm_vector)

    def shift_matrix(self):
        """Shifts the matrix diagonal by the Froebenius norm to make all eigenvalues positive. That is,
        if A is the matrix of the system and ||A||_F is the Froebenius norm, this method does the shift

                A --> A + ||A||_F * I =: A'

        where I is the identity matrix of the same dimension as A.

        This transformation ensures all eigenvalues of A' are non-negative.

        Note: If the matrix has already been shifted (i.e., if this method has already been called), then
        calling this method again will do nothing.

        Modifies:
            The matrix of QSVE and the BinaryTrees.
        """
        # If the matrix is already shifted, do nothing
        if self._shifted:
            return

        # Compute the Froebenius norm
        norm = self.matrix_norm()

        # Shift each diagonal entry, updating both the tree and matrix
        for (diag, tree) in enumerate(self._trees):
            # Get the current value
            value = self._matrix[diag][diag]

            # Update the matrix
            self._matrix[diag][diag] = value + norm

            # Update the BinaryTree
            tree.update_entry(diag, value + norm)

        # Set the shifted flag to True
        self._shifted = True

    @staticmethod
    def _controlled_reflection_circuit(circuit, ctrl_qubit, register):
        """Adds the gates for a controlled reflection about the |0> state to the input circuit.
        This circuit does the reflection I - 2|0><0| where I is the identity gate.


        This circuit has the following structure:

                    qubit       ------------@------------
                                            |
                                ------------O------------
                                            |
                                ------------O------------
                    register                |
                                ------------O------------
                                            |
                                ----[X]----[Z]----[X]----

        where @ represents a control on |1> and O represents a control on |0>.

        Note: This can also be done with an ancilla qubit using phase kickback. The circuit above avoids the need
        for an ancilla qubit.

        Note: In the circuit construction, the identity HXH = Z is utilized to use a multi-controlled NOT gate instead
        of a multi-controlled Z gate.

        Args:
            circuit : qiskit.QuantumCircuit
                The QuantumCircuit object to add gates to.
                This circuit must contain both the ctrl_qubit and the register.

            ctrl_qubit : qiskit.QuantumRegister.qubit
                The qubit to control the reflection on. This qubit must be in the input circuit.

            register : qiskit.QuantumRegister
                The register to perform the reflection on.

        Returns:
            None

        Modifies:
            Input circuit. Adds gates to this circuit to perform the controlled reflection.
        """
        # Input argument checks
        if type(circuit) != QuantumCircuit:
            raise TypeError(
                "Argument circuit must be of type qiskit.QuantumCircuit."
            )

        if ctrl_qubit not in circuit.qubits:
            raise ValueError(
                "Argument ctrl_qubit must be in circuit.qubits."
            )

        if register not in circuit.qregs:
            raise ValueError(
                "Argument register must be in circuit.qregs."
            )

        # Add NOT gates on all qubits in the reflection register (for anti-controls)
        circuit.x(register)

        # Add a Hadamard on the last qubit in the reflection register for phase kickback
        circuit.h(register[-1])

        # Add the multi-controlled NOT (Tofolli) gate
        mct(circuit, [ctrl_qubit] + register[:-1], register[-1], None, mode="noancilla")

        # Add a Hadamard on the last qubit in the reflection register for phase kickback
        circuit.h(register[-1])

        # Add NOT gates on all qubits in the reflection registers (for anti-controls)
        circuit.x(register)

    def controlled_unitary(self, circuit, qpe_qubit, row_register, col_register):
        """Adds the gates for one Controlled-W unitary to the input circuit.

        The input circuit must have at least four registers corresponding to the input arguments

        At a high-level, this circuit has the following structure:

                    QPE (q qubit)   -------@--------
                                           \
                    ROW (n qubits)  ----|      |----
                                        |      |
                    COL (m qubits)  ----|  W   |----
                                        |      |
                    PKB (1 qubit)   ----|______|----

        At a lower level, the controlled-W circuit is implemented as follows:

            QPE (1 qubit)   ---------------------@------------------------------@-----------------
                                                 |                              |
            ROW (n qubits)  ----| V^dagger |----[R]----| V |---|           |----|----|   |--------
                                                               | W^dagger  |    |    | W |
            COL (m qubits)  -----------------------------------|           |---[R]---|   |--------

        where @ is a control symbol and O is an "anti-control" symbol (i.e., controlled on the |0> state).
        The gate R is a reflection about the |0> state.

        TODO: Add "mathematical section" explaining what V and W are.

        Args:
            circuit : qiskit.QuantumCircuit
                The QuantumCircuit object that gates will be added to.
                This QuantumCircuit must have at least three registers, enumerated below.
                Any gates already in the circuit are un-modified. The gates to implement Controlled-W are added after
                these gates.

            qpe_qubit : qiskit.QuantumRegister
                Quantum register used for precision in phase estimation. In the diagrams above, this is labeled QPE.
                The number of qubits in this register (p) is chosen by the user.

            row_register : qiskit.QuantumRegister
                Quantum register used to load/store rows of the matrix. In the diagrams above, this is labeled ROW.
                The number of qubits in this register (m) must be m = log2(number of matrix rows).

            col_register : qiskit.QuantumRegister
                Quantum register used to load/store columns of the matrix. In the diagrams above, this is labeled COL.
                The number of qubits in this register (n) must be n = log2(number of matrix cols).

        Returns:
            None

        Modifies:
            The input circuit. Adds gates to this circuit to implement the controlled-W unitary.
        """
        # =====================
        # Check input arguments
        # =====================

        if type(circuit) != QuantumCircuit:
            raise TypeError(
                "The argument circuit must be of type qiskit.QuantumCircuit."
            )

        if len(circuit.qregs) < 3:
            raise ValueError(
                "The input circuit does not have enough quantum registers."
            )

        if len(row_register) != len(col_register):
            raise ValueError(
                "Only square matrices are currently supported." +
                "This means the row_register and col_register must have the same number of qubits."
            )

        if len(row_register) != self._num_qubits_for_row:
            raise ValueError(
                "Invalid number of qubits for row_register. This number should be {}".format(self._num_qubits_for_row)
            )

        if qpe_qubit not in circuit.qubits:
            raise ValueError(
                "Argument qpe_qubit must be in circuit.qubits."
            )

        for register in (row_register, col_register):
            if register not in circuit.qregs:
                raise ValueError(
                    "The input circuit has no register {}.".format(register)
                )

        # =======================================================================================
        # Store a copy of the circuit with all gates removed for the controlled row loading gates
        # =======================================================================================

        ctrl_row_load_circuit = deepcopy(circuit)
        ctrl_row_load_circuit.data = []

        # =================
        # Build the circuit
        # =================

        # Get the row norm circuit
        row_norm_circuit = self.row_norm_tree.preparation_circuit(row_register)

        # Add the inverse row norm circuit. This corresponds to V^dagger in the doc string circuit diagram.
        circuit += row_norm_circuit.inverse()

        # Add the controlled reflection on the row register. This corresponds to
        # the first C(R) in the doc string circuit diagram.
        self._controlled_reflection_circuit(circuit, qpe_qubit, row_register)

        # Add the row norm circuit. This corresponds to V in the doc string diagram.
        circuit += row_norm_circuit

        # # DEBUG
        # circuit.barrier()

        # Get the controlled row loading operations. This corresponds to W in the doc string circuit diagram.
        for ii in range(self.matrix_ncols):
            row_tree = self.get_tree(ii)

            # Add the controlled row loading circuit
            ctrl_row_load_circuit += row_tree.preparation_circuit(
                row_register, control_register=col_register, control_key=ii
            )

        # Add W^dagger to the circuit
        circuit += ctrl_row_load_circuit.inverse()

        # Add the controlled reflection on the column register. This corresponds to
        # the second C(R) in the doc string circuit diagram.
        self._controlled_reflection_circuit(circuit, qpe_qubit, col_register)

        # Add W to the circuit
        circuit += ctrl_row_load_circuit

        # # DEBUG
        # circuit.barrier()

    @staticmethod
    def _qft(circuit, register, final_swaps=False):
        """Adds the gates for a quantum Fourier Transform (QFT) to the input circuit in the specified register.

        Args:
            circuit : qiskit.QuantumCircuit
                Circuit to add gates to.

            register : qiskit.QuantumRegister
                Register in the circuit where the QFT is performed.

        Returns:
            None

        Modifies:
            circuit. Adds the gates for the QFT.
        """
        # Error checking on input arguments
        if type(circuit) != QuantumCircuit:
            raise TypeError(
                "Argument circuit must be of type qiskit.QuantumCircuit."
            )

        if register not in circuit.qregs:
            raise ValueError(
                "Argument register must be a valid qiskit.QuantumRegister in circuit.qregs."
            )

        # Get the number of qubits in the register
        nqubits = len(register)

        # Add the gates
        for targ in range(nqubits - 1, -1, -1):
            print("targ =", targ)
            # Add the Hadamard gate
            circuit.h(register[targ])

            # Add the controlled Rz gates
            for ctrl in range(targ - 1, -1, -1):
                angle = - 2 * np.pi * 2**(ctrl - targ)
                circuit.cu1(angle, register[ctrl], register[targ])

        if final_swaps:
            for qubit in range(nqubits // 2):
                ctrl = qubit
                targ = nqubits - ctrl - 1
                circuit.swap(register[ctrl], register[targ])

    def phase_estimation(self, circuit, qpe_register, row_register, col_register):
        """Adds the phase estimation subroutine to the input circuit.

        Args:
            circuit : qiskit.QuantumCircuit
                The QuantumCircuit object that gates will be added to.
                This QuantumCircuit must have at least three registers, enumerated below.
                Any gates already in the circuit are un-modified. The gates to implement QPE are added after
                these gates.

            qpe_register : qiskit.QuantumRegister
                Quantum register used for precision in phase estimation.
                The number of qubits in this register (p) is chosen by the user.

            row_register : qiskit.QuantumRegister
                Quantum register used to load/store rows of the matrix.
                The number of qubits in this register (m) must be m = log2(number of matrix rows).

            col_register : qiskit.QuantumRegister
                Quantum register used to load/store columns of the matrix.
                The number of qubits in this register (n) must be n = log2(number of matrix cols).


        See help(QSVE.controlled_unitary) for further details on these registers.

        Returns:
            None

        Modifies:
            circuit
        """
        # Do the round of Hadamards on the precision register
        circuit.h(qpe_register)

        # # DEBUG
        # circuit.barrier()

        # Do the controlled unitary operators
        for (p, qpe_qubit) in enumerate(qpe_register):
            for _ in range(2**p):
                self.controlled_unitary(circuit, qpe_qubit, row_register, col_register)

        # DEBUG
        circuit.barrier()

        # TODO: Do the QFT on the precision register
        self._qft(circuit, qpe_register)

    def create_circuit(self):
        """Returns a quantum circuit implementing the QSVE algorithm (without cosine)."""
        # Create the quantum registers
        qpe_register = QuantumRegister(self._num_qubits_for_qpe)
        row_register = QuantumRegister(self._num_qubits_for_row)
        col_register = QuantumRegister(self._num_qubits_for_col)

        # Create the quantum circuit
        circuit = QuantumCircuit(qpe_register, row_register, col_register)

        # TODO: Do the optional state preparation

        # Do phase estimation
        self.phase_estimation(circuit, qpe_register, row_register, col_register)

        return circuit


if __name__ == "__main__":
    matrix = np.random.randn(4, 4)
    matrix += matrix.conj().T

    qsve = QSVE(matrix, nprecision_bits=3)

    print(qsve.matrix)

    print()

    print(qsve.row_norm_tree)
    print(qsve.get_tree(0))
    print(qsve.get_tree(1))

    circ = qsve.create_circuit()

    # print(circ)

    print("\n\nBase circuit stats:\n")

    print("Depth = ", circ.depth())
    print("# gates =", sum(circ.count_ops().values()))
    print("# qubits =", len(circ.qubits))
    print(circ.count_ops())

    trans = transpile(circ, optimization_level=2)

    print("\n\nTranspiled circuit stats:\n")

    print("Depth = ", trans.depth())
    print("# gates =", sum(trans.count_ops().values()))
    print("# qubits =", len(trans.qubits))
    print(trans.count_ops())

    print("\n\nDecomposed circuit stats:\n")

    dec = circ.decompose()

    print("Depth = ", dec.depth())
    print("# gates =", sum(dec.count_ops().values()))
    print("# qubits =", len(dec.qubits))
    print(dec.count_ops())

    print("\n\nTranspiled decomposed circuit stats:\n")

    circ = transpile(dec, optimization_level=2)

    print("Depth = ", circ.depth())
    print("# gates =", sum(circ.count_ops().values()))
    print("# qubits =", len(circ.qubits))
    print(circ.count_ops())
