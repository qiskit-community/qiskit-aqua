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
from itertools import permutations

import numpy as np


from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits.gates import mct
from qiskit.aqua.circuits.gates.controlled_ry_gates import mcry
from qiskit import QuantumRegister, QuantumCircuit, execute, BasicAer


class BinaryTree:
    """Binary tree data structure used to store and access matrix elements in QSVD."""
    def __init__(self, vector):
        """Initializes a BinaryTree.

        Args:
            vector : array-like
                Array of values in one row of a matrix.
        """
        # Store the number of values in the matrix row
        self._nvals = len(vector)

        # Make sure the matrix row has length that's a power of two
        if self._nvals & (self._nvals - 1) != 0:
            raise ValueError(
                "Matrix row must have a number of elements that is a power of two. " +
                "Please append zero entries to the row."
            )

        # If the L2 norm of the matrix row isn't unity, normalize it
        vector /= np.linalg.norm(vector, ord=2)

        # Store the input matrix_row
        self._values = list(vector)
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

    def get_leaf(self, index):
        """Returns the indexed element.

        Args:
            index : int
                Index of leaf element to return.

        Return type: float.
        """
        return self._values[index]

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
        if level == 0:
            return None

        return (level - 1, index // 2)

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

        return (level + 1, 2 * index)

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

        return (level + 1, 2 * index + 1)

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

    def preparation_circuit(self, register):
        """Returns a circuit that encodes the leaf values (input values to the BinaryTree)in a quantum state.

        For example, if the vector

        [0.4, 0.4, 0.8, 0.2]

        is input to BinaryTree, then this method returns a circuit which prepares the state

        0.4|00> + 0.4|01> + 0.8|10> + 0.2|11>.

        This circuit consists of controlled-Y rotations. It can optionally be controlled on qubits (see args below).

        Args:
            register : QuantumRegister
                Quantum register to prepare the circuit in.

        """
        # Error checks: Make sure the register has enough qubits
        if int(2**len(register)) < self._nvals:
            raise ValueError(
                "Not enough qubits in input register to store vector."
            )

        # TODO: Why not create the register in the method?

        # Get the number of ancilla qubits needed to do the multi-controlled-Y rotations
        if len(register) <= 3:
            circ = QuantumCircuit(register)
        else:
            num_ancillae = len(register) - 3
            ancilla_register = QuantumRegister(num_ancillae)
            circ = QuantumCircuit(register, ancilla_register)

        # First rotation angle
        parent = self.root
        left_child = self.left_child_value(0, 0)
        theta = 2 * np.arccos(np.sqrt(left_child / parent))

        # Do the Y-rotation
        circ.ry(theta, register[0])

        # Special case for getting the sign correct with only a single qubit
        if self.number_levels == 2:
            if self._values[0] < 0:
                circ.x(register[0])
                circ.z(register[0])
                circ.x(register[0])
            if self._values[1] < 0:
                circ.z(register[0])

        # =========================================
        # Traverse the tree and add the Y-rotations
        # =========================================

        # Loop down the levels of the tree, starting at the first (after the root)
        for level in range(1, self.number_levels - 1):
            # Within this level, loop from left to right across nodes
            for index in range(len(self._tree[level])):
                # Get the index of the node in binary
                bitstring = np.binary_repr(index, level)

                # Get the rotation angle
                parent = self._tree[level][index]
                left_child = self.left_child_value(level, index)

                # Don't divide by zero. Note that parent == 0 implies both children are 0,
                #  so theta = 0.0 is indeed the correct angle to rotate by.
                if np.isclose(parent, 0.0):
                    continue
                else:
                    theta = 2 * np.arccos(np.sqrt(left_child / parent))

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

                # Do X gates for anti-controls
                for (ii, bit) in enumerate(bitstring):
                    if bit == "0":
                        circ.x(register[ii])

                # =========================================
                # Do the Multi-Controlled-Y (MCRY) rotation
                # =========================================

                # For three qubits or less, no ancilla are needed to do the MCRY
                if len(register) <= 3:
                    if mct_flag:
                        mct(circ, register[:level], register[level], None)
                        mct_flag = False
                    mcry(circ, theta, register[:level], register[level], None)
                # For more than three qubits, ancilla are needed
                else:
                    if mct_flag:
                        mct(circ, register[:level], register[level], ancilla_register)
                        mct_flag = False
                    mcry(circ, theta, register[:level], register[level], ancilla_register)

                # Do X gates for anti-controls
                for (ii, bit) in enumerate(bitstring):
                    if bit == "0":
                        circ.x(register[ii])

        return circ

    def __str__(self):
        """Returns a formatted string representation of the tree."""
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
                rowstring += elt  # Note: "".join(row) does not format correctly here.
            rowstring += "\n"
            string += rowstring
        return string


class QSVE(CircuitFactory):
    pass


# ==========
# Unit tests
# ==========

def test_basic():
    """Basic checks for a BinaryTree."""
    # Instantiate a BinaryTree
    tree = BinaryTree([1, 1])

    # Simple checks
    assert np.isclose(tree.root, 1.0)
    assert tree.number_leaves == 2
    assert tree.number_levels == 2


def test_example_in_paper():
    """Tests correctness for the binary tree in the example given in
    Appendix A of the quantum recommendation systems paper.
    """
    # The same vector used in the paper
    row = [0.4, 0.4, 0.8, 0.2]

    # Construct the tree from the vector
    tree = BinaryTree(row)

    # Make sure the elements are equal
    assert np.isclose(tree.data[0][0], 1.0)
    assert np.isclose(tree.data[1][0], 0.32)
    assert np.isclose(tree.data[1][1], 0.68)
    assert np.isclose(tree.data[2][0], 0.16)
    assert np.isclose(tree.data[2][1], 0.16)
    assert np.isclose(tree.data[2][2], 0.64)
    assert np.isclose(tree.data[2][3], 0.04)


def test_print_small():
    """Tests the correct string format is obtained when printing a small tree."""
    tree = BinaryTree([1, 1])
    correct = "    1.00    \n0.50    0.50\n"
    assert tree.__str__() == correct


def test_print_medium():
    """Tests the correct string format is obtained when printing a tree with four leaves."""
    tree = BinaryTree([1, 1, 1, 1])
    correct = "        1.00        \n    0.50            0.50    \n0.25    0.25    0.25    0.25\n"
    assert tree.__str__() == correct


def test_number_leaves():
    """Tests the number of leaves is correct for a BinaryTree."""
    # Create a tree
    tree = BinaryTree(np.ones(128))

    # Make sure the number of leaves is correct
    assert tree.number_leaves == 128


def test_number_levels():
    """Tests the number of levels is correct for a BinaryTree."""
    # Create a tree
    tree = BinaryTree(np.ones(32))

    # Make sure the number of leaves is correct
    assert tree.number_levels == 6


def test_parent_indices():
    """Tests correctness for getting parent indices.

    The relevant indexing structure here is:

                     (0, 0)
                       ^
               (1, 0)     (1, 1)
                 ^          ^
           (2, 0) (2, 1) (2, 2) (2, 3)

    """
    tree = BinaryTree([1, 1, 1, 1])

    # Test that the parent of root is none
    assert tree.parent_index(0, 0) is None

    # Test that the parent's of the first level are the root
    assert tree.parent_index(1, 0) == (0, 0)
    assert tree.parent_index(1, 1) == (0, 0)

    # Test that the parent's of the second level are correct
    assert tree.parent_index(2, 0) == (1, 0)
    assert tree.parent_index(2, 1) == (1, 0)
    assert tree.parent_index(2, 2) == (1, 1)
    assert tree.parent_index(2, 3) == (1, 1)


def test_parent_value():
    """Tests correctness for getting the parent value of a node."""
    # Get a BinaryTree
    tree = BinaryTree([1, 1, 1, 1])

    # Make sure the parent of the root is None
    assert tree.parent_value(0, 0) is None

    # Make sure the parent's of the first level (root) are correct
    assert np.isclose(tree.parent_value(1, 0), 1.0)
    assert np.isclose(tree.parent_value(1, 1), 1.0)

    # Make sure the parent's of the second level are correct
    assert np.isclose(tree.parent_value(2, 0), 0.50)
    assert np.isclose(tree.parent_value(2, 1), 0.50)
    assert np.isclose(tree.parent_value(2, 2), 0.50)
    assert np.isclose(tree.parent_value(2, 3), 0.50)


def test_left_child_index():
    """Tests getting the index of the left child."""
    # Get a BinaryTree
    tree = BinaryTree([1, 2, 3, 4])

    # Left child of the root
    assert tree.left_child_index(0, 0) == (1, 0)

    # Left child indices for the first level
    assert tree.left_child_index(1, 0) == (2, 0)
    assert tree.left_child_index(1, 1) == (2, 2)

    # Left child indices for leaves
    assert tree.left_child_index(2, 0) is None
    assert tree.left_child_index(2, 1) is None
    assert tree.left_child_index(2, 2) is None
    assert tree.left_child_index(2, 3) is None


def test_right_child_index():
    """Tests getting the index of a right child for a set of nodes."""
    # Get a BinaryTree
    tree = BinaryTree([1, 2, 3, 4])

    # Right child of the root
    assert tree.right_child_index(0, 0) == (1, 1)

    # Right child indices for the first level
    assert tree.right_child_index(1, 0) == (2, 1)
    assert tree.right_child_index(1, 1) == (2, 3)

    # Right child indices for leaves
    assert tree.right_child_index(2, 0) is None
    assert tree.right_child_index(2, 1) is None
    assert tree.right_child_index(2, 2) is None
    assert tree.right_child_index(2, 3) is None


def test_left_child_value():
    """Tests getting the value of a left child for a set of nodes."""
    # Get a BinaryTree
    tree = BinaryTree([1, 1, 1, 1])

    # Check the root value
    assert np.isclose(tree.root, 1.0)

    # Left child of the root
    assert np.isclose(tree.left_child_value(0, 0), 0.5)

    # Left child indices for the first level
    assert np.isclose(tree.left_child_value(1, 0), 0.25)
    assert np.isclose(tree.left_child_value(1, 1), 0.25)

    # Left child indices for leaves
    assert tree.left_child_value(2, 0) is None
    assert tree.left_child_value(2, 1) is None
    assert tree.left_child_value(2, 2) is None
    assert tree.left_child_value(2, 3) is None


def test_right_child_value():
    """Tests getting the value of a right child for a set of nodes."""
    # Get a BinaryTree
    tree = BinaryTree([1, 1, 1, 1])

    # Check root value
    assert np.isclose(tree.root, 1.0)

    # Right child of the root
    assert np.isclose(tree.right_child_value(0, 0), 0.5)

    # Right child indices for the first level
    assert np.isclose(tree.right_child_value(1, 0), 0.25)
    assert np.isclose(tree.right_child_value(1, 1), 0.25)

    # Right child indices for leaves
    assert tree.right_child_value(2, 0) is None
    assert tree.right_child_value(2, 1) is None
    assert tree.right_child_value(2, 2) is None
    assert tree.right_child_value(2, 3) is None


def final_state(circuit):
    """Returns the final state of the circuit as a numpy.ndarray."""
    # Get the unitary simulator backend
    sim = BasicAer.get_backend("unitary_simulator")

    # Execute the circuit
    job = execute(circuit, sim)

    # Get the final state
    unitary = np.array(job.result().results[0].data.unitary)
    return unitary[:, 0]


def test_prep_circuit_one_qubit():
    """Tests for correctness in preparation circuit on a single qubit."""
    # Two element vector
    vec = [1.0, 0.0]

    # Make a BinaryTree
    tree = BinaryTree(vec)

    # Get the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(1))

    # Get the final state
    state = list(np.real(final_state(circ)))

    assert np.array_equal(state, vec)


def test_prep_circuit_one_qubit2():
    """Tests for correctness in preparation circuit on a single qubit."""
    # Two element vector
    vec = [0.0, 1.0]

    # Make a BinaryTree
    tree = BinaryTree(vec)

    # Get the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(1))

    # Get the final state
    state = list(np.real(final_state(circ)))

    assert np.array_equal(state, vec)


def test_prep_circuit_one_qubit3():
    """Tests for correctness in preparation circuit on a single qubit."""
    # Two element vector
    vec = [0.6, 0.8]

    # Make a BinaryTree
    tree = BinaryTree(vec)

    # Get the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(1))

    # Get the final state
    state = list(np.real(final_state(circ)))

    assert np.array_equal(state, vec)


def test_prep_circuit_example_in_paper():
    """Tests the state preparation circuit produces the correct state
    for the example given in the quantum recommendations system paper.
    """
    # The same vector used in the paper
    vec = [0.4, 0.4, 0.8, 0.2]

    # Construct the tree from the vector
    tree = BinaryTree(vec)

    # Construct the state preparation circuit
    qreg = QuantumRegister(2)
    circuit = tree.preparation_circuit(qreg)

    # Add a swaps to make the ordering of the qubits match the input vector
    # Note: This is because the last bit is the most significant in qiskit, not the first.
    circuit.swap(qreg[0], qreg[1])

    # Check that the circuit produces the correct state
    state = list(np.real(final_state(circuit)))
    assert np.allclose(state, vec)


def test_prep_circuit_three_qubits():
    """Tests the state preparation circuit produces the correct state on three qubits."""
    # Input vector (normalized)
    vec = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Get the state preparation circuit
    qreg = QuantumRegister(3)
    circuit = tree.preparation_circuit(qreg)

    # Add a swaps to make the ordering of the qubits match the input vector
    # Note: This is because the last bit is the most significant in qiskit, not the first.
    circuit.swap(qreg[0], qreg[2])

    # Check that the circuit produces the correct state
    state = list(np.real(final_state(circuit)))
    assert np.allclose(state, vec)


def test_prep_circuit_medium():
    """Tests the state preparation circuit produces the correct state for a moderate number of qubits."""
    # Input vector (normalized)
    vec = np.ones(16)
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Do the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(4))

    # Check that the circuit produces the correct state
    # Note: No swaps are necessary here since all amplitudes are equal.
    state = np.real(final_state(circ))

    # Note: The output state has an additional ancilla needed to do the multi-controlled-Y rotations,
    # so we discard the additional (zero) amplitudes when comparing to the input vector
    assert np.allclose(state[:16], vec)


def test_prep_circuit_large():
    """Tests the state preparation circuit produces the correct state for many qubits."""
    # Input vector (normalized)
    vec = np.ones(64)
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Do the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(6))

    # Check that the circuit produces the correct state
    # Note: No swaps are necessary here since all amplitudes are equal.
    state = np.real(final_state(circ))

    # Note: The output state has an additional ancilla needed to do the multi-controlled-Y rotations,
    # so we discard the additional (zero) amplitudes when comparing to the input vector
    assert np.allclose(state[:64], vec)


def test_prep_circuit_large2():
    """Tests the state preparation circuit produces the correct state for many qubits."""
    # Input vector (normalized)
    vec = np.array(list(np.ones(32)) + list(np.zeros(32)))
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Do the state preparation circuit
    qreg = QuantumRegister(6)
    circ = tree.preparation_circuit(qreg)

    # Do the swaps to get the ordering of amplitudes to match with the input vector
    for ii in range(len(qreg) // 2):
        circ.swap(qreg[ii], qreg[-ii - 1])

    # Check that the circuit produces the correct state
    state = np.real(final_state(circ))

    # Note: The output state has an additional ancilla needed to do the multi-controlled-Y rotations,
    # so we discard the additional (zero) amplitudes when comparing to the input vector
    assert np.allclose(state[:64], vec)


def test_prepare_negative_amplitudes():
    """Tests preparing a vector with negative amplitudes on a single qubit."""
    vec = [0.6, -0.8]

    tree = BinaryTree(vec)

    circuit = tree.preparation_circuit(QuantumRegister(1))

    state = np.real(final_state(circuit))

    assert np.allclose(state, vec)


def test_prepare_negative_amplitudes2():
    """Tests preparing a vector with negative amplitudes on a single qubit."""
    vec = [-0.6, 0.8]

    tree = BinaryTree(vec)

    circuit = tree.preparation_circuit(QuantumRegister(1))

    state = np.real(final_state(circuit))

    assert np.allclose(state, vec)


def test_prepare_negative_amplitudes3():
    """Tests preparing a vector with negative amplitudes on a single qubit."""
    vec = [-0.6, -0.8]

    tree = BinaryTree(vec)

    circuit = tree.preparation_circuit(QuantumRegister(1))

    state = np.real(final_state(circuit))

    assert np.allclose(state, vec)


def test_prepare_negative_amplitudes_two_qubits():
    """Tests preparing a vector with negative amplitudes for the example from
    the quantum recommendations systems paper.
    """
    vec = [-0.4, 0.4, -0.8, 0.2]

    tree = BinaryTree(vec)

    qreg = QuantumRegister(2)

    circuit = tree.preparation_circuit(qreg)

    circuit.swap(qreg[0], qreg[1])

    state = np.real(final_state(circuit))

    assert np.allclose(state, vec)


def test_prepare_negative_amplitudes_two_qubits2():
    """Tests preparing a vector with negative amplitudes on a single qubit."""
    # Generate all sign configurations
    one_neg = set(permutations((-1, 1, 1, 1)))
    two_neg = set(permutations((-1, -1, 1, 1)))
    three_neg = set(permutations((-1, -1, -1, 1)))
    four_neg = {(-1, -1, -1, -1)}

    for sign in one_neg | two_neg | three_neg | four_neg:
        # Input vector (normalized)
        vec = np.array([-1, -2, 3, 4], dtype=np.float64)
        vec *= np.array(sign, dtype=np.float64)
        vec /= np.linalg.norm(vec, ord=2)

        tree = BinaryTree(vec)

        qreg = QuantumRegister(2)

        circuit = tree.preparation_circuit(qreg)

        circuit.swap(qreg[0], qreg[1])

        state = np.real(final_state(circuit))

        assert np.allclose(state, vec)


def test_prepare_negative_amplitudes_three_qubits():
    """Tests state preparation for a vector on three qubits with negative amplitudes."""
    # Input vector (normalized)
    vec = np.array([-1, -2, 3, -4, -5, 6, -7, 8], dtype=np.float64)
    vec /= np.linalg.norm(vec, ord=2)

    # Get the BinaryTree
    tree = BinaryTree(vec)

    # Quantum register
    qreg = QuantumRegister(3)

    # Get the state preparation circuit
    circuit = tree.preparation_circuit(qreg)

    # Add swaps to compare amplitudes with normal vector ordering
    circuit.swap(qreg[0], qreg[2])

    # Make sure the final state is equal to the input vector
    state = np.real(final_state(circuit))
    assert np.allclose(state, vec)


def test_prep_circuit_negative_amplitudes_large():
    """Tests the state preparation circuit produces the correct state for many qubits."""
    # Input vector (normalized)
    vec = -1.0 * np.ones(64)
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Do the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(6))

    # Check that the circuit produces the correct state
    # Note: No swaps are necessary here since all amplitudes are equal.
    state = np.real(final_state(circ))

    # Note: The output state has an additional ancilla needed to do the multi-controlled-Y rotations,
    # so we discard the additional (zero) amplitudes when comparing to the input vector
    assert np.allclose(state[:64], vec)


def test_prepare_negative_amplitudes_four_qubits():
    """Tests state preparation for a vector on three qubits with negative amplitudes."""
    # Input vector (normalized)
    vec = np.array([-1, -2, 3, -4, -5, 6, -7, 8, 9, 10, 11, -12, -13, -14, 15, -16], dtype=np.float64)
    vec /= np.linalg.norm(vec, ord=2)

    # Get the BinaryTree
    tree = BinaryTree(vec)

    # Quantum register for input
    qreg = QuantumRegister(4)

    # Create the state preparation circuit
    circuit = tree.preparation_circuit(qreg)

    # Add swaps to compare with normal vector ordering
    circuit.swap(qreg[0], qreg[3])
    circuit.swap(qreg[1], qreg[2])

    # Get the final state of the circuit
    state = np.real(final_state(circuit))

    # Only compare the first 16 amplitudes (ancillae are needed to do multi-controlled gates)
    assert np.allclose(state[:16], vec)


def test_prep_circuit_large():
    """Tests the state preparation circuit produces the correct state for many qubits."""
    # Input vector (normalized)
    vec = np.ones(64)
    vec /= np.linalg.norm(vec, ord=2)

    # Make a tree from the vector
    tree = BinaryTree(vec)

    # Do the state preparation circuit
    circ = tree.preparation_circuit(QuantumRegister(6))

    # Check that the circuit produces the correct state
    # Note: No swaps are necessary here since all amplitudes are equal.
    state = np.real(final_state(circ))

    # Note: The output state has an additional ancilla needed to do the multi-controlled-Y rotations,
    # so we discard the additional (zero) amplitudes when comparing to the input vector
    assert np.allclose(state[:64], vec)


if __name__ == "__main__":
    test_basic()
    test_example_in_paper()
    test_print_small()
    # test_print_medium()
    test_number_leaves()
    test_number_levels()
    test_parent_indices()
    test_parent_value()
    test_left_child_index()
    test_right_child_index()
    test_left_child_value()
    test_right_child_value()
    test_prep_circuit_one_qubit()
    test_prep_circuit_one_qubit2()
    test_prep_circuit_one_qubit3()
    test_prep_circuit_example_in_paper()
    test_prep_circuit_three_qubits()
    test_prep_circuit_medium()
    test_prep_circuit_large()
    test_prep_circuit_large2()
    test_prepare_negative_amplitudes()
    test_prepare_negative_amplitudes2()
    test_prepare_negative_amplitudes3()
    test_prepare_negative_amplitudes_two_qubits()
    test_prepare_negative_amplitudes_two_qubits2()
    test_prepare_negative_amplitudes_three_qubits()
    test_prepare_negative_amplitudes_four_qubits()
    test_prep_circuit_negative_amplitudes_large()
