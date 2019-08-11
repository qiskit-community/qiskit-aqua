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

"""Module for Quantum Singular Value Estimation (QSVE), and algorithm for
estimating the singular values of a Hermitian matrix.

QSVE is used as a subroutine in quantum recommendation systems [1],
where it was first introduced, and in linear systems solvers [2].

QSVE performs quantum phase estimation with a unitary whose eigenvalues
of which are related to the singular values of A.

For more details and a tutorial, see

https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua

References:

    [1] I. Kerenidis and A. Prakash, “Quantum Recommendation Systems,”
        arXiv:1603.08675 [quant-ph], Mar. 2016.

    [2] L. Wossnig, Z. Zhao, and A. Prakash, “A quantum linear system
        algorithm for dense matrices,” Phys. Rev. Lett., vol. 120, no. 5,
        p. 050502, Jan. 2018.
"""

# Imports
from copy import deepcopy
import operator
import numpy as np
import warnings

from qiskit.aqua.circuits.gates.multi_control_toffoli_gate import mct
from qiskit.aqua.components.qsve import BinaryTree
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, BasicAer, transpile
from qiskit.aqua.components.initial_states import Custom


class MatrixError(Exception):
    pass


class QSVE:
    """Quantum Singular Value Estimation (QSVE) class."""
    def __init__(self, matrix):
        """Initializes a QSVE object.

        Args:
            matrix : numpy.ndarray
                The matrix to perform singular value estimation on.
        """
        # Make sure the matrix is of the correct type
        if not isinstance(matrix, np.ndarray):
            raise MatrixError("Input matrix must be of type numpy.ndarray.")

        if len(matrix.shape) != 2:
            raise MatrixError("Input matrix must be two-dimensional.")

        # Get the number of rows and columns in the matrix
        nrows, ncols = matrix.shape

        # Make sure the number of rows is supported
        if nrows & (nrows - 1) != 0:
            raise MatrixError("Number of rows in matrix must be a power of two.")

        # Make sure the number of columns is supported
        if ncols & (ncols - 1) != 0:
            raise MatrixError("Number of columns in matrix must be a power of two.")

        # Store these as attributes
        self._matrix_nrows = nrows
        self._matrix_ncols = ncols

        # Store the number of qubits needed for the matrix rows and cols
        self._num_qubits_for_row = int(np.log2(nrows))
        self._num_qubits_for_col = int(np.log2(ncols))

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

    @property
    def nprecision_bits(self):
        """Returns the number of precision bits, i.e., the number of qubits used in the phase estimation register."""
        return self._nprecision_bits

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

    def _row_norm_vector(self):
        """Returns a vector of the row norms of the input matrix."""
        # Initialize an empty list for the vector of row norms
        row_norm_vector = []

        # Append all the row norms
        for tree in self._trees:
            row_norm_vector.append(np.sqrt(tree.root))

        return np.array(row_norm_vector)

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
        # Return the BinaryTree made from the row norm vector
        return BinaryTree(self._row_norm_vector())

    def shift_matrix(self, fraction=1.0):
        """Shifts the matrix diagonal by the Froebenius norm to make all eigenvalues positive. That is,
        if A is the matrix of the system and ||A||_F is the Froebenius norm, this method does the shift

                A --> A + fraction * ||A||_F * I =: A'

        where I is the identity matrix of the same dimension as A.

        This transformation ensures all eigenvalues of A' are non-negative.

        Args:
            fraction : float
                Floating point value to multiply the Froebenius norm by when performing the shift. (See above equation.)

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

    def row_isometry(self):
        """Returns the row isometry (U) used to build the unitary for QPE.

        Return type:
            numpy.ndarray
        """
        # Dimension of the input matrix A
        dim = self._matrix_nrows

        # Dimension of the U isometry
        umat = np.zeros((dim**2, dim))

        # Fill out the columns of U
        for col in range(dim):
            basis = np.zeros(dim)
            basis[col] = 1
            umat[:, col] = np.kron(basis, self._matrix[col, :]) / np.linalg.norm(self._matrix[col, :], ord=2)

        return umat

    def norm_isometry(self):
        """Returns the norm isometry (V) used to build the unitary for QPE.

        Return type:
            numpy.ndarray
        """
        # Dimension of the input matrix A
        dim = self._matrix_nrows

        # Dimension of the V isometry
        vmat = np.zeros((dim**2, dim))

        # Get the vector of row norms
        norms = self._row_norm_vector()

        # Fill out the columns of U
        for col in range(dim):
            basis = np.zeros(dim)
            basis[col] = 1
            vmat[:, col] = np.kron(norms, basis)

        return vmat / self.matrix_norm()

    def unitary(self):
        """Returns a classical (matrix) representation of the unitary W(A).

        This unitary is used in phase estimation to compute (approximate) singular values.

        Return type:
            numpy.ndarray
        """
        # Get the row isometry and reflection
        umat = self.row_isometry()
        uref = 2 * umat @ umat.conj().T - np.identity(self._matrix_nrows**2)

        # Get the norm isometry
        vmat = self.norm_isometry()
        vref = 2 * vmat @ vmat.conj().T - np.identity(self._matrix_nrows**2)

        return uref @ vref

    def unitary_evecs(self):
        """Returns the eigenvectors of the unitary W = (2U U^dagger - I)(2V V^dagger -I) used for phase estimation.

        This is useful for providing an initial state for phase estimation.

        Return type:
            numpy.ndarray
        """
        _, evecs = np.linalg.eig(self.unitary())
        return evecs

    def singular_vectors_classical(self):
        """Returns the singular vectors of the matrix for QSVE found classically.

        Return type:
            numpy.ndarray
        """
        vecs, _, _ = np.linalg.svd(self._matrix)
        return vecs

    def singular_values_classical(self, normalized=True):
        """Returns the singular values of the matrix for QSVE found classically.

        Return type:
            numpy.ndarray
        """
        sigmas = np.linalg.svd(self._matrix, compute_uv=False)
        if normalized:
            return sigmas / self.matrix_norm()
        return sigmas

    @staticmethod
    def _controlled_reflection_circuit(circuit, ctrl_qubit, *registers):
        """Adds the gates for a controlled reflection about the |0> state to the input circuit.
        This circuit does the reflection I - 2|0><0| where I is the identity gate.


        This circuit has the following structure:

                    qubit       ------------@------------
                                            |
                                ------------O------------
                                            |
                                ------------O------------
                    registers               |
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

            registers : qiskit.QuantumRegister
                The register(s) to perform the reflection on.

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

        if isinstance(ctrl_qubit, QuantumRegister):
            if len(ctrl_qubit) > 1:
                raise ValueError(
                    "There is more than one qubit in the argument ctrl_qubit." +
                    "Either pass a single qubit (e.g., register[0]) or a register of only one qubit."
                )
            elif len(ctrl_qubit) == 1:
                ctrl_qubit = ctrl_qubit[0]

        if ctrl_qubit not in circuit.qubits:
            raise ValueError(
                "Argument ctrl_qubit must be in circuit.qubits."
            )

        all_qubits = []
        for register in registers:
            if register not in circuit.qregs:
                raise ValueError(
                    "Argument register must be in circuit.qregs."
                )
            all_qubits += register[:]

        # Add NOT gates on all qubits in the reflection register (for anti-controls)
        circuit.x(all_qubits)

        # Add a Hadamard on the last qubit in the reflection register for phase kickback
        circuit.h(all_qubits[-1])

        # Add the multi-controlled NOT (Tofolli) gate
        mct(circuit, [ctrl_qubit] + all_qubits[:-1], all_qubits[-1], None, mode="noancilla")

        # Add a Hadamard on the last qubit in the reflection register for phase kickback
        circuit.h(all_qubits[-1])

        # Add NOT gates on all qubits in the reflection registers (for anti-controls)
        circuit.x(all_qubits)

    def controlled_row_norm_reflection(self, circuit, ctrl_qubit, *registers):
        """Implements the controlled reflection I - 2 M M^dagger where M is the isometry corresponding to row norms of
        the input matrix.

        Args:
            circuit : qiskit.QuantumCircuit
                Circuit in which to implement the controlled row norm reflection.

            ctrl_qubit : qiskit.QuantumRegister.qubit
                Qubit to control the reflection on.

            *registers : qiskit.QuantumRegister(s)
                Variable number of registers in which to prepare states for implementing the isometry M.

        Returns:
            None

        Modifies:
            circuit
                Adds gates to implement the controlled reflection.
        """
        row_load_circuit = deepcopy(circuit)
        row_load_circuit.data = []

        # Get the row norm circuit
        self.row_norm_tree.preparation_circuit(row_load_circuit, *registers)

        # Add the inverse row norm circuit. This corresponds to V^dagger in the doc string circuit diagram.
        circuit += row_load_circuit.inverse()

        # Add the controlled reflection on the row register. This corresponds to
        # the first C(R) in the doc string circuit diagram.
        self._controlled_reflection_circuit(circuit, ctrl_qubit, *registers)

        # Add the row norm circuit. This corresponds to V in the doc string diagram.
        circuit += row_load_circuit

    def controlled_row_reflection(self, circuit, ctrl_qubit, *registers):
        """Implements the controlled reflection I - 2 N N^dagger where N is the isometry corresponding to row vectors
        of the input matrix.

                Args:
                    circuit : qiskit.QuantumCircuit
                        Circuit in which to implement the controlled row norm reflection.

                    ctrl_qubit : qiskit.QuantumRegister.qubit
                        Qubit to control the reflection on.

                    *registers : qiskit.QuantumRegister(s)
                        Variable number of registers in which to prepare states for implementing the isometry N.

                Returns:
                    None

                Modifies:
                    circuit
                        Adds gates to implement the controlled reflection.
                """

        ctrl_row_load_circuit = deepcopy(circuit)
        ctrl_row_load_circuit.data = []

        # Get the controlled row loading operations. This corresponds to W in the doc string circuit diagram.
        for ii in range(self.matrix_nrows):
            row_tree = self.get_tree(ii)

            # Add the controlled row loading circuit
            row_tree.preparation_circuit(
                ctrl_row_load_circuit, registers[1], control_register=registers[0], control_key=ii
            )

        # Add W^dagger to the circuit
        circuit += ctrl_row_load_circuit.inverse()

        # Add the controlled reflection on the column register. This corresponds to
        # the second C(R) in the doc string circuit diagram.
        self._controlled_reflection_circuit(circuit, ctrl_qubit, registers[1])

        # Add W to the circuit
        circuit += ctrl_row_load_circuit

    def controlled_unitary(self, circuit, qpe_qubit, row_register, col_register):
        """Adds the gates for one Controlled-W unitary to the input circuit.

        The input circuit must have at least three registers corresponding to the input arguments (more below).

        At a high-level, this circuit has the following structure:

                    QPE (q qubit)   -------@--------
                                           \
                    ROW (n qubits)  ----|      |----
                                        |  W   |
                    COL (m qubits)  ----|      |----


        At a lower level, the controlled-W circuit is implemented as follows:

            QPE (1 qubit)   ---------------------@------------------------------@-----------------
                                                 |                              |
            ROW (n qubits)  ----| V^dagger |----[R]----| V |---|           |----|----|   |--------
                                                               | U^dagger  |    |    | U |
            COL (m qubits)  -----------------------------------|           |---[R]---|   |--------

        where @ is a control symbol and O is an "anti-control" symbol (i.e., controlled on the |0> state).
        The gate R is a reflection about the |0> state, and U, V are isometries. (See [1] for full details.)

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

        References:
            [1] I. Kerenidis and A. Prakash, “Quantum Recommendation Systems,”
                arXiv:1603.08675 [quant-ph], Mar. 2016.
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

        # ==============================================================================================
        # Store a copies of the circuit for (controlled) row loading. This makes it easy to get inverses
        # ==============================================================================================

        ctrl_row_load_circuit = deepcopy(circuit)
        ctrl_row_load_circuit.data = []

        # =================
        # Build the circuit
        # =================

        # Add the gates for the controlled row norm reflection
        self.controlled_row_norm_reflection(circuit, qpe_qubit, row_register)

        # Add the gates for the controlled row reflection
        self.controlled_row_reflection(circuit, qpe_qubit, row_register, col_register)

    @staticmethod
    def _iqft(circuit, register, final_swaps=False):
        """Adds gates for the inverse quantum Fourier Transform (IQFT) to the input circuit in the specified register.

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
            # Add the Hadamard gate
            circuit.h(register[targ])

            # Add the controlled Rz gates
            for ctrl in range(targ - 1, -1, -1):
                angle = - 2 * np.pi * 2**(ctrl - targ - 1)
                circuit.cu1(angle, register[ctrl], register[targ])

        if final_swaps:
            for qubit in range(nqubits // 2):
                ctrl = qubit
                targ = nqubits - ctrl - 1
                circuit.swap(register[ctrl], register[targ])

    def phase_estimation(self, circuit, qpe_register, row_register, col_register, logical_barriers=False):
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

            logical_barriers : bool
                If True, barriers are added to the circuit to visually separate W, W^2, W^4, ...

        See help(QSVE.controlled_unitary) for further details on these registers.

        Returns:
            None

        Modifies:
            circuit
        """
        # Do the round of Hadamards on the precision register
        circuit.h(qpe_register)

        # Add a barrier, if desired
        if logical_barriers:
            circuit.barrier()

        # Do the controlled unitary operators
        for (p, qpe_qubit) in enumerate(qpe_register):
            for _ in range(2**p):
                self.controlled_unitary(circuit, qpe_qubit, row_register, col_register)

            # Add a barrier, if desired
            if logical_barriers:
                circuit.barrier()

        # Add the inverse QFT on the precision register
        self._iqft(circuit, qpe_register)

    def _prepare_singular_vector(self, singular_vector, circuit, *registers):
        """Prepares the singular vector on the input register.

        Args:
            singular_vector : Union[numpy.ndarray, qiskit.QuantumCircuit]
                The singular vector to prepare in the "column register," which can be input as a vector (numpy array)
                or a quantum circuit which prepares the vector.

            circuit : qiskit.QuantumCircuit
                Circuit to prepare the singular vector in.

            registers : qiskit.QuantumRegister
                Register(s) to prepare the singular vector in.
        """
        if singular_vector is None:
            return

        all_qubits = []
        for reg in registers:
            all_qubits += reg[:]

        if isinstance(singular_vector, (np.ndarray, list)):
            # If all vector elements are real, we can use the binary tree to load the vector
            if all(np.isreal(singular_vector)):
                tree = BinaryTree(singular_vector)
                tree.preparation_circuit(circuit, *registers)
            # If some vector elements are complex, use custom state preparation
            else:
                state = Custom(
                    num_qubits=len(all_qubits), state_vector=singular_vector
                )
                circuit += state.construct_circuit(register=all_qubits)

        elif type(singular_vector) == QuantumCircuit:
            if len(singular_vector.qregs) != 2:
                raise ValueError("If singular_vector is input as a circuit, there must be exactly two registers " +
                                 "in the order 'row' then 'col'. See help(QSVE.controlled_unitary) for more info.")
            warnings.warn(
                "There must be two registers named 'row' and 'col' in the singular_vector circuit, "
                "else undefined behavior will occur."
            )
            circuit += singular_vector

        else:
            raise ValueError(
                "Singular vector must be input as a numpy.ndarray, list," +
                "or a quantum circuit preparing the desired vector."
            )

    def create_circuit(
            self,
            nprecision_bits=3,
            init_state_row_and_col=None,
            load_row_norms=False,
            init_state_col=None,
            terminal_measurements=False,
            return_registers=False,
            logical_barriers=False,
            **kwargs
    ):
        """Returns a quantum circuit implementing the QSVE algorithm.

        Note: The output of this circuit is not the singular values but angles theta related to the singular values
        sigma by
                                        cos(theta * pi) = sigma / ||A||_F


        Args:
            nprecision_bits : int (default value = 3)
                The number of qubits to use in phase estimation.
                Equivalently, the number of bits of precision to read out singular values.

            init_state_row_and_col : Union[numpy.ndarray, qiskit.QuantumCircuit]
                A vector to prepare across the row and column registers.
                This can be input as a vector (numpy array) or a quantum circuit which prepares the vector.

            load_row_norms : bool
                If True, the row norm vector is prepared in the row register.
                This is useful for applications of QSVE, namely linear systems and recommendation systems.

            init_state_col : Union[numpy.ndarray, qiskit.QuantumCircuit]
                A vector to prepare in the column register.
                This is useful for applications of QSVE, namely linear systems and recommendation systems.

            terminal_measurements : bool (default: False)
                If True, measurements are added to the phase register at the end of the circuit, else nothing happens.

            return_registers : bool (default: False)
                If True, registers are returned along with the circuit.
                Note: Registers can be accessed from the circuit -- this option is for convenience.

                The order of the returned registers is:
                    (1) QPE Register: Qubits used for precision in QPE.
                    (2) ROW Register: Qubits used for loading row norms of the matrix.
                    (3) COL Register: Qubits used for loading rows of the matrix.

            logical_barriers : bool (default: False)
                If True, barriers are inserted in the circuit between logical components (subroutines).

        Returns : qiskit.QuantumCircuit
            The quantum circuit with gates implementing the QSVE algorithm.

            If return_registers==True, then the registers in the above circuit are returned as well.
            Note that these registers can be accessed from the circuit itself. This option is for convenience.
        """
        # Parse the register names from keyword arguments, if provided
        qpe_register_name = kwargs["qpe_name"] if "qpe_name" in kwargs.keys() else "qpe"
        row_register_name = kwargs["row_name"] if "row_name" in kwargs.keys() else "row"
        col_register_name = kwargs["col_name"] if "col_name" in kwargs.keys() else "col"

        # Create the quantum registers
        qpe_register = QuantumRegister(nprecision_bits, name=qpe_register_name)
        row_register = QuantumRegister(self._num_qubits_for_row, name=row_register_name)
        col_register = QuantumRegister(self._num_qubits_for_col, name=col_register_name)

        # Create the quantum circuit
        circuit = QuantumCircuit(qpe_register, row_register, col_register)

        # Add the optional state preparation in the column register
        if init_state_row_and_col is not None:
            self._prepare_singular_vector(init_state_row_and_col, circuit, row_register, col_register)

            # Add a barrier, if desired
            if logical_barriers:
                circuit.barrier()

        # Load the row norms of the matrix in the column register
        if load_row_norms:
            self.row_norm_tree.preparation_circuit(circuit, row_register)

            # Add a barrier, if desired
            if logical_barriers:
                circuit.barrier()

        if init_state_col is not None:
            self._prepare_singular_vector(init_state_col, circuit, col_register)

            # Add a barrier, if desired
            if logical_barriers:
                circuit.barrier()

        # Do phase estimation
        self.phase_estimation(circuit, qpe_register, row_register, col_register, logical_barriers)

        # Add a barrier, if desired
        if logical_barriers:
            circuit.barrier()

        if terminal_measurements:
            creg = ClassicalRegister(nprecision_bits)
            circuit.add_register(creg)
            circuit.measure(qpe_register, creg)

        if return_registers:
            return circuit, qpe_register, row_register, col_register
        return circuit

    def run_and_return_counts(
            self,
            nprecision_bits=3,
            init_state_row_and_col=None,
            shots=10000,
            ordered=True
    ):
        """Creates the quantum circuit for QSVE with terminal measurements and executes it, returning the counts.

        Args:
            nprecision_bits : int
                Number of qubits to use for QPE.

            init_state_row_and_col : Union[list, numpy.ndarray, None]
                Initial state to start the row and column register in for phase estimation.

            shots : int
                Number of times to execute the circuit.

            ordered : bool
                If True, the returned measurement outcomes are ordered from most to least frequent.

        Returns : list<tuple<str, int>>
            List of tuples of the form [(bitstring1, counts1), (bitsring2, counts2), ...]
        """
        # Create the circuit with terminal measurements
        circuit = self.create_circuit(
            nprecision_bits,
            init_state_row_and_col=init_state_row_and_col,
            terminal_measurements=True
        )

        # Get a simulator
        sim = BasicAer.get_backend("qasm_simulator")
        job = execute(circuit, sim, shots=shots)

        # Get the output bit strings from QSVE
        res = job.result()
        counts = res.get_counts()
        if ordered:
            counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        return counts

    def top_singular_values(
            self,
            nprecision_bits=3,
            init_state_row_and_col=None,
            shots=10000,
            ntop=1
    ):
        """Returns the top estimated singular value(s) from the QSVE algorithm.

        Args:
            nprecision_bits : int
                Number of precision qubits to use in QPE.

            init_state_row_and_col : Union[list, numpy.ndarray, None]
                Initial state to start the row and column register in for phase estimation.

            shots : int
                Number of times to execute the circuit.

            ntop : int
                Number of top singular values to return. Note: To return all, set ntop=-1.

        Returns : list
            List of `ntop` normalized singular values (floats) estimated by the quantum circuit.
        """
        # Get the ordered counts
        counts = self.run_and_return_counts(
            nprecision_bits,
            init_state_row_and_col=init_state_row_and_col,
            shots=shots,
            ordered=True
        )

        # Get the top counts
        top = [count[0] for count in counts[:ntop]]

        # Convert the bit strings to floating point values in the range [0, 1)
        values = [self.binary_decimal_to_float(bits) for bits in top]

        # Convert the measured values to angles theta in the range [-1/2, 1/2)
        thetas = [self.convert_measured(val) for val in values]

        # Convert the floating point values to singular values
        qsigmas = [self.angle_to_singular_value(theta) for theta in thetas]

        return qsigmas

    def has_value_close_to_singular_values(self, sigmas, tolerance):
        """Returns True if at least one value in the list sigmas is close to a (normalized) singular value.

        Args:
            sigmas : list<float>
                List of floating point values.

        Returns : bool
            True or False, as above.
        """
        correct = self.singular_values_classical()
        for sigma in sigmas:
            for val in correct:
                if abs(sigma - val) < tolerance:
                    return True
        return False

    def expected_raw_outcome(self, nbits, shots, init_state):
        evals = np.linalg.eig(self.unitary())

        thetas = [self.to_binary_decimal(self.unitary_eval_to_angle(evalue), nbits) for evalue in evals]

        counts = {}

        for theta in thetas:
            if theta not in counts.keys():
                counts[theta] = 1
            else:
                counts[theta] += 1

        # TODO: Sample from the distribution

    @staticmethod
    def possible_estimated_singular_values(nprecision_bits):
        step = -2**(-nprecision_bits)
        measured = np.arange(0.5, 0.0 + step, step)
        return np.array([np.cos(np.pi * theta) for theta in measured])

    @staticmethod
    def unitary_eval_to_angle(evalue):
        """Returns the angle theta such that e^(i theta) = evalue.

        Args:
            evalue: complex
                Complex eigenvalue with modulus one.

        Returns: float
            Angle theta in the interval [0, 1).
        """
        if not np.isclose(abs(evalue), 1.0):
            raise ValueError("Invalid eigenvalue (Modulus isn't one.)")
        return np.arccos(np.real(evalue)) / 2 / np.pi

    @staticmethod
    def angle_to_singular_value(theta):
        """Returns the singular value for the given angle.

        Args:
            theta : float
                Angle such that 0 <= theta <= 1.
        """
        return np.cos(np.pi * theta)

    @staticmethod
    def unitary_eval_to_singular_value(evalue):
        return QSVE.angle_to_singular_value(QSVE.unitary_eval_to_angle(evalue))

    @staticmethod
    def binary_decimal_to_float(binary_decimal, big_endian=False):
        """Returns a floating point value from an input binary decimal represented as a string.

        Args:
            binary_decimal : str
                String representing a binary decimal.

            big_endian : bool
                If True, the most significant bit in the binary_decimal is first.
                If False, the most significant bit in the binary_decimal is last.

                Examples:
                    "01" with big_endian == True is 0.01 = 0.25 (1/4).
                    "01" with big_endian == False is 0.10 = 0.50 (1/2).

        Returns: float
            Floating point value represented by the binary string.
        """
        if not big_endian:
            binary_decimal = reversed(binary_decimal)

        val = 0.0
        for (ii, bit) in enumerate(binary_decimal):
            if bit == "1":
                val += 2 ** (-ii - 1)
        return val

    @staticmethod
    def convert_measured(theta):
        """Converts a measured angle theta (float) in the interval [0, 1) from QPE to a value in the range [-1/2, 1/2].

        Args:
            theta : float
                Floating point value in the range [0, 1).
                (The floating point value of the binary decimal measured in QPE.)
        """
        if 1.0 > theta < 0.0:
            raise ValueError("Argument theta must satisfy 0 <= theta <= 1, but theta = {}.".format(theta))
        if 0.0 <= theta <= 0.5:
            return theta
        else:
            return theta - 1.0

    @staticmethod
    def max_error(nbits):
        """Returns the maximum possible error on sigma / ||A||_F given the input number of precision bits."""
        if nbits == 1:
            return 0.5
        cosines = np.array([np.cos(np.pi * k / 2**nbits) for k in range(2**(nbits - 1))])
        return max(abs(np.diff(cosines))) / 2

    @staticmethod
    def to_binary_decimal(decimal, nbits=5):
        """Converts a decimal in base ten to a binary decimal string.

        Args:
            decimal : float
                Floating point value in the interval [0, 1).

            nbits : int
                Number of bits to use in the binary decimal.

        Return type:
            str
        """
        if decimal < 0 or decimal >= 1:
            raise ValueError("Argument decimal should satisfy 0 <= decimal < 1.")

        binary = ""
        while len(binary) < nbits:
            jj = 1
            while decimal * (2**jj) < 1 and len(binary) < nbits:
                binary += "0"
                jj += 1
            if len(binary) == nbits:
                return binary
            binary += "1"
            decimal = (decimal * 2**jj) % 1
        return binary
