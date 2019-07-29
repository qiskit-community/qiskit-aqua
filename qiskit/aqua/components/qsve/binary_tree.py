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

"""Module containing the definition of a BinaryTree."""

# Imports
from copy import deepcopy

import numpy as np

from qiskit.aqua.circuits.gates.multi_control_toffoli_gate import mct
from qiskit.aqua.circuits.gates.multi_control_rotation_gates import mcry
from qiskit import QuantumRegister, QuantumCircuit


class BinaryTree:
    """Binary tree data structure used for loading an input vector onto a quantum state."""
    def __init__(self, vector):
        """Initializes a BinaryTree.

        Args:
            vector : array-like
                Array of values in one row of a matrix.
        """
        # Make a copy of the vector
        vector = deepcopy(vector)

        # Store the number of values in the matrix row
        self._nvals = len(vector)

        # Make sure the matrix row has length that's a power of two
        # TODO: Give the option to pad the vector and do this automatically
        if self._nvals & (self._nvals - 1) != 0:
            raise ValueError(
                "Matrix row must have a number of elements that is a power of two. " +
                "Please append zero entries to the row."
            )

        # Store the input vector
        self._values = list(vector)
        self._vector = np.array(self._values)
        self._nvals = len(self._values)

        # ===========================================================
        # Construct the tree upside down (and then reverse the order)
        # ===========================================================

        # Store the sign information
        self._tree = [list(map(lambda x: np.sign(x), self._values))]

        # Store the magnitude squared values
        self._tree.append(list(map(lambda x: abs(x) ** 2, self._values)))

        # Sum adjacent elements to build the next row of the tree
        while len(self._tree[-1]) > 1:
            vals = []
            for ii in range(0, len(self._tree[-1]) - 1, 2):
                vals.append(self._tree[-1][ii] + self._tree[-1][ii + 1])
            self._tree.append(vals)

        # Reverse the order of the tree
        self._tree = list(reversed(self._tree))

    @property
    def data(self):
        """The data structure (list of lists) storing the binary tree.

        Return type: list<list>.
        """
        return self._tree

    @property
    def root(self):
        """Returns the root of the tree.

        Return type: float.
        """
        return self._tree[0][0]

    @property
    def number_leaves(self):
        """The number of leaves in the tree, equal to the length of the input vector.
        (Number of elements in last level of tree.)
        """
        return int(self._nvals)

    @property
    def number_levels(self):
        """The number of levels in the tree."""
        return int(np.ceil(np.log2(self._nvals)) + 1)

    @property
    def leaves(self):
        return self._tree[-2]

    def get_level(self, level):
        """Returns a level in the tree.

        Examples:
            level = 0
                Returns the root of the tree.

            level = 1
                Returns a list of the two nodes below the root.

            etc.

        Return type: list
        """
        return self._tree[level]

    def get_element(self, level, index):
        """Returns an element in the tree.

        Args:
            level : int
                Level of the tree the element is in.

            index : int
                Index of the element in the level.

        Return type: float
        """
        return self._tree[level][index]

    def parent_index(self, level, index):
        """Returns the indices of the parent of a specified node.

        Args:
            level : int
                The node's level in the tree.

            index : int
                The node's index within a level.

        Return type: tuple<int, int>.
        """
        if level == 0 or level > self.number_levels:
            return None

        return level - 1, index // 2

    def parent_value(self, level, index):
        """Returns the value of the parent of a specified node.

                Args:
                    level : int
                        The node's level in the tree.

                    index : int
                        The node's index within a level.

                Return type: float.
                """
        # Check if root node
        if level == 0:
            return None

        # Get the level and index of the parent
        level, index = self.parent_index(level, index)

        # Return the parent value
        return self._tree[level][index]

    def left_child_index(self, level, index):
        """Returns the index of the left child of a specified parent node.

        Args:
            level : int
                The parent node's level in the tree.

            index : int
                The parent node's index within the level.

        Return type: tuple<int, int>.
        """
        if level == self.number_levels - 1:
            return None

        return level + 1, 2 * index

    def right_child_index(self, level, index):
        """Returns the index of the right child of a specified parent node.

            Args:
                level : int
                    The parent node's level in the tree.

                index : int
                    The parent node's index within the level.

            Return type: tuple<int, int>.
            """
        if level == self.number_levels - 1:
            return None

        return level + 1, 2 * index + 1

    def left_child_value(self, level, index):
        """Returns the value of the left child of a specified parent node.

            Args:
                level : int
                    The parent node's level in the tree.

                index : int
                    The parent node's index within the level.

            Return type: float.
            """
        if level == self.number_levels - 1:
            return None

        level, index = self.left_child_index(level, index)

        return self._tree[level][index]

    def right_child_value(self, level, index):
        """Returns the value of the right child of a specified parent node.

            Args:
                level : int
                    The parent node's level in the tree.

                index : int
                    The parent node's index within the level.

            Return type: float.
            """
        if level == self.number_levels - 1:
            return None

        level, index = self.right_child_index(level, index)

        return self._tree[level][index]

    def update_entry(self, index, value):
        """Updates an entry in the leaf and propagates changes up through the tree."""
        # TODO: Do this more efficiently.
        newvals = self._values
        newvals[index] = value
        self.__init__(newvals)

    def preparation_circuit(self, register, control_register=None, control_key=None, use_ancillas=False):
        """Returns a circuit that encodes the leaf values (input values to the BinaryTree)in a quantum state.

        For example, if the vector [0.4, 0.4, 0.8, 0.2] is input to BinaryTree, then this method returns a circuit
        which prepares the state

        0.4|00> + 0.4|01> + 0.8|10> + 0.2|11>.

        This circuit consists of controlled-Y rotations, and has the following structure for two qubits:

                 |0> ----[Ry(theta1)]----@---------------O-----------
                                         |               |
                 |0> ---------------[Ry(theta2)]---[Ry(theta3)]------


        Here, the @ symbol represents a control and the O symbol represents an "anti-control" (controlled on |1> state).

        All gates can optionally be controlled on another control_register. See arguments below.

        Args:
            register : qiskit.QuantumRegister
                The state of this register will be the vector of the BinaryTree.

            control_register : qiskit.QuantumRegister
                Every gate added to the register will be controlled on all qubits in this register.

            control_key : Union[int, str]
                The control_key determines which qubits in the control_register are "anti-controls" or regular controls.

                For example, if the control_register has two qubits, the possible values for control_key are:

                        int     | str       | meaning
                        ==============================
                        0       | "00"      | Control on both qubits
                        1       | "01"      | Control on the first qubit, anti-control on the second qubit.
                        2       | "10"      | Anti-control on the first qubit, control on the second qubit.
                        3       | "11"      | Anti-control on both qubits.

                An example circuit for control_key = 1 is shown schematically below:

                        preparation_circuit(reg, ctrl_reg, 1) -->

                                    |  -------@-------
                        ctrl_reg    |         |
                                    |  -------O-------
                                              |
                                    |  ----|     |----
                        reg         |  ----|     |----
                                    |  ----|_____|----

                An example for control_key = 2 is shown schematically below:

                        preparation_circuit(reg, ctrl_reg, 2) -->

                                    |  -------O-------
                        ctrl_reg    |         |
                                    |  -------@-------
                                              |
                                    |  ----|     |----
                        reg         |  ----|     |----
                                    |  ----|_____|----

            use_ancillas : bool (default value = False)
                Flag to determine whether or not to use ancillas for multi-controlled gates.
                Using ancillas can reduce the number of gates.
        """
        # =========================
        # Checks on input arguments
        # =========================

        # Make sure the register has enough qubits
        if int(2**len(register)) < self._nvals:
            raise ValueError(
                "Not enough qubits in input register to store vector. " +
                "A register with at least {} qubits is needed.".format(int(np.log2(self._nvals)))
            )

        # Make sure a control_key is provided if a control_register is provided
        if control_register is not None:
            if control_key is None:
                raise ValueError("If control_register is provided, a valid control_key must also be provided.")

        # Make sure a control_register is provided if the control_key is provided
        if control_key is not None:
            if control_register is None:
                raise ValueError("If control_key is provided, a valid control_register must also be provided.")

        if control_register:
            if type(control_register) != QuantumRegister:
                raise ValueError("Argument control_register must be of type qiskit.QuantumRegister.")

            # Get the number of qubits and dimension of the control register
            num_control_qubits = len(control_register)
            max_control_key = 2**num_control_qubits

        if control_key is not None:
            if not isinstance(control_key, (int, str)):
                raise ValueError("Argument control_key must be of type int or str.")

            # If provided as an integer
            if type(control_key) == int:
                if 0 > control_key > max_control_key:
                    raise ValueError(
                        "Invalid integer value for control_key." +
                        "This argument must be in the range [0, 2^(len(control_register))."
                    )

                # Convert to a string
                control_key = np.binary_repr(control_key, num_control_qubits)

            # If provided as a string
            if type(control_key) == str:
                if len(control_key) != num_control_qubits:
                    raise ValueError(
                        "Invalid string value for control_key. " +
                        "The control_key must have the same number of characters as len(control_register)"
                    )

        # =======================
        # Get the quantum circuit
        # =======================

        # Get the base quantum circuit
        circ = QuantumCircuit(register)

        # =================================================================
        # Special case: Computational basis vector with no control register
        # =================================================================

        if not control_register and self._is_basis_vector(self._vector):
            self._prepare_basis_vector(circ, register, self._vector)
            return circ

        # ======================================================
        # Add control registers and ancilla registers, if needed
        # ======================================================

        # Add the control register if provided
        # Note: the technique for adding controls for every gate will be adding the control_register_qubits to
        # a list of controlled qubits. If empty, nothing changes. If non-empty, we get the correct controls.
        if control_register:
            circ.add_register(control_register)
            control_register_qubits = control_register[:]
        else:
            control_register_qubits = []

        # Add ancilla qubits, if necessary, to do multi-controlled Y rotations
        if use_ancillas and len(register) > 3:
            # TODO: Figure out exactly how many ancillae are needed.
            # TODO: Take into account the length of control_register_qubits above
            if control_register:
                num_ancillae = max(len(register) - 1, 3)
            else:
                num_ancillae = max(len(register) + len(control_register) - 1, 3)
            ancilla_register = QuantumRegister(num_ancillae)
            circ.add_register(ancilla_register)

        # ================================================================
        # Special case: Computational basis vector with a control register
        # ================================================================

        if control_register and self._is_basis_vector(self._vector):
            self._prepare_basis_vector_control(circ, register, control_register, control_key, self._vector)
            return circ

        # ============================
        # Add the gates to the circuit
        # ============================

        # Do the initial pattern of X gates on control register (if provided) to get controls & anti-controls correct
        if control_register_qubits:
            for (ii, bit) in enumerate(control_key):
                if bit == "0":
                    circ.x(control_register[ii])

        # =========================================
        # Traverse the tree and add the Y-rotations
        # =========================================

        # Loop down the levels of the tree, starting at the first level (below the root)
        for level in range(0, self.number_levels - 1):
            # Within this level, loop from left to right across nodes
            for index in range(len(self._tree[level])):
                # Get the index of the node in binary
                bitstring = np.binary_repr(index, level)

                # Get the rotation angle
                parent = self._tree[level][index]
                left_child = self.left_child_value(level, index)

                # Avoid empty nodes
                if np.isclose(parent, 0.0):
                    continue

                # If the right child is zero, the gate is identity ==> do nothing.
                if np.isclose(self.right_child_value(level, index), 0.0):
                    continue

                # Compute the angle
                theta = 2 * np.arccos(np.sqrt(left_child / parent))

                # Initialize flag to perform a CNOT. The CNOT is used if both amplitudes are negative.
                mct_flag = False

                # If we're on the last row, shift the angle to take sign information into account
                if level == self.number_levels - 2:
                    # Get the actual (not squared) value of the left leaf child
                    left_child_leaf_index = 2 * index
                    left_child_leaf_value = self._values[left_child_leaf_index]

                    # Get the actual (not squared) value of the right leaf child
                    right_child_leaf_index = 2 * index + 1
                    right_child_leaf_value = self._values[right_child_leaf_index]

                    # Get the appropriate angle
                    if left_child_leaf_value < 0.0 and right_child_leaf_value > 0.0:
                        theta = 2 * np.arcsin(np.sqrt(left_child / parent)) + np.pi

                    elif left_child_leaf_value > 0.0 and right_child_leaf_value < 0.0:
                        theta = 2 * np.arcsin(np.sqrt(left_child / parent)) - np.pi

                    elif left_child_leaf_value < 0.0 and right_child_leaf_value < 0.0:
                        theta = 2 * np.arccos(np.sqrt(left_child / parent)) + np.pi

                        # Set flag to do the controlled bit flip on the target of the MCRY gate
                        mct_flag = True

                # =========================================
                # Do the Multi-Controlled-Y (MCRY) rotation
                # =========================================

                # Do X gates for anti-controls on the state preparation register
                if level > 0:
                    for (ii, bit) in enumerate(bitstring):
                        if bit == "0":
                            if control_register_qubits:
                                mct(circ, control_register_qubits, register[ii], None, mode="noancilla")
                            else:
                                circ.x(register[ii])

                # Get all control qubits
                if level == 0:
                    all_control_qubits = control_register_qubits
                else:
                    all_control_qubits = control_register_qubits + register[:level]

                # For three qubits or less, no ancilla are needed to do the MCRY
                if len(register) <= 3 or not use_ancillas:
                    # Do the CNOT for the special case of both amplitudes negative
                    if mct_flag:
                        if len(all_control_qubits) > 0:
                            mct(circ, all_control_qubits, register[level], None, mode="noancilla")
                        else:
                            circ.x(register[level])

                    # Do the Y-rotation
                    if len(all_control_qubits) > 0:
                        mcry(circ, theta, all_control_qubits, register[level], None, mode="noancilla")
                    else:
                        circ.ry(theta, register[level])

                # For more than three qubits, ancilla are needed
                else:
                    # Do the CNOT for the special case of both amplitudes negative
                    if mct_flag:
                        if len(all_control_qubits) > 0:
                            mct(circ, all_control_qubits, register[level], ancilla_register)
                        else:
                            circ.x(register[level])

                    # Do the Y-rotation
                    if len(all_control_qubits) > 0:
                        mcry(circ, theta, all_control_qubits, register[level], ancilla_register)
                    else:
                        circ.ry(theta, register[level])

                # Do X gates for anti-controls on the state preparation register
                if level > 0:
                    for (ii, bit) in enumerate(bitstring):
                        if bit == "0":
                            if control_register_qubits:
                                mct(circ, control_register_qubits, register[ii], None, mode="noancilla")
                            else:
                                circ.x(register[ii])

        # Do the final pattern of X gates on the control qubits to get controls & anti-controls correct
        if control_register_qubits:
            for (ii, bit) in enumerate(control_key):
                if bit == "0":
                    circ.x(control_register[ii])

        return circ

    @staticmethod
    def _is_basis_vector(vector):
        """Returns True if the vector is proportional to a computational basis vector."""
        return len(vector.nonzero()[0]) == 1

    @staticmethod
    def _prepare_basis_vector(circuit, register, vector):
        """Prepares a computational basis vector from the |0> state."""
        # Determine which element is nonzero
        elt = vector.nonzero()[0][0]

        # Convert the index to a binary string
        bitstring = np.binary_repr(elt, len(register))

        # Add NOT gates in the right places
        for (ii, bit) in enumerate(bitstring):
            if bit == "1":
                circuit.x(register[ii])

    @staticmethod
    def _prepare_basis_vector_control(circuit, register, control_register, control_key, vector):
        """Prepares a computational basis vector from the |0> state controlled on a control register."""
        # Determine which element is nonzero
        elt = vector.nonzero()[0][0]

        # Convert the index to a binary string
        bitstring = np.binary_repr(elt, len(register))

        # Do the initial pattern of X gates on control register to get controls & anti-controls correct
        for (ii, bit) in enumerate(control_key):
            if bit == "0":
                circuit.x(control_register[ii])

        # Add NOT gates in the right places
        for (ii, bit) in enumerate(bitstring):
            if bit == "1":
                mct(circuit, control_register, register[ii], None, mode="noancilla")

        # Do the final pattern of X gates on control register to get controls & anti-controls correct
        for (ii, bit) in enumerate(control_key):
            if bit == "0":
                circuit.x(control_register[ii])

    def __str__(self):
        """Returns a string representation of the tree."""
        # Get the shape of the array
        shape = (int(np.ceil(np.log2(self._nvals)) + 1), 2 * self._nvals - 1)

        # Initialize an empty array
        arr = np.empty(shape, dtype=object)

        # Returns the step (distance between elements) in the xth row of the array
        step = lambda x: 2**(x + 1)

        # Returns the skip (horizontal offset) in the xth row of the array
        skip = lambda x: 2**x - 1

        # Loop through the tree and store the values as formatted strings
        for ii in range(len(self._tree) - 1):
            for (jj, val) in enumerate(self._tree[ii]):
                # Format the value
                if np.isclose(val, 0.0):
                    string = "    "
                else:
                    string = "%0.2f" % val

                # Get the correct column index
                col = len(self._tree) - ii - 2
                col_index = skip(col) + step(col) * jj

                # Put the string in the array
                arr[ii][col_index] = string

        # Replace None objects with string separators.
        # Note: To format correctly, the number of spaces must be the same as
        # the number of chars in each value, which is by default 4.
        arr[arr == None] = "    "

        # Format the array as a string
        string = ""
        for row in arr:
            rowstring = ""
            for elt in row:
                rowstring += elt
            rowstring += "\n"
            string += rowstring
        return string
