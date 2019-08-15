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

from qiskit.aqua.components.qsve import QSVE
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.aqua.circuits.gates.multi_control_toffoli_gate import mct
from qiskit.aqua import QuantumAlgorithm
from qiskit.aqua.utils.decimal_to_binary import decimal_to_binary


class UserVectorError(Exception):
    pass


class ThresholdError(Exception):
    pass


class RankError(Exception):
    pass


class QuantumRecommendation(QuantumAlgorithm):
    """Class for a quantum recommendation system."""

    CONFIGURATION = {
        'name': 'QuantumRecommendation',
        'description': 'Quantum Recommendation Systems algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qrs_schema',
            'type': 'object',
        },
        'problems': ["recommend"]
    }

    def __init__(self, preference_matrix, user_vector, threshold, nprecision_bits=3):
        """Initializes a QuantumRecommendation.

        Args:
            preference_matrix : numpy.ndarray

            user_vector : numpy.ndarray

            threshold : float

            nprecision_bits : int
                Number of precision qubits to use in QPE.
        """
        super().__init__()
        self._matrix = deepcopy(preference_matrix)
        self._user = deepcopy(user_vector)
        self._threshold_value = threshold
        self._precision = nprecision_bits
        self._qsve = QSVE(preference_matrix)
        self._ret = {}

    @property
    def matrix(self):
        return self._matrix

    @property
    def num_users(self):
        return len(self._matrix)

    @property
    def num_products(self):
        return np.shape(self._matrix)[1]

    def _threshold(self, circuit, register, ctrl_string, measure_flag_qubit=True):
        """Adds gates to the input circuit to perform a thresholding operation, which discards singular values
        below some threshold, or minimum value.

        Args:
            circuit : qiskit.QuantumCircuit
                Circuit to the thresholding operation (circuit) to.

            register : qiskit.QuantumRegister
                A register (in the input circuit) to threshold on. For recommendation systems, this will always be
                the "singular value register."

            ctrl_string : str
                A binary string which determines the number of qubits controlled on in the singular value register.
                See circuit below.

        The thresholding circuit adds a register of three ancilla qubits to the circuit and has the following structure:

                            |0> --------O--------@-----------------------@--------O--------
                                        |        |                       |        |
                            |0> --------O--------@-----------------------@--------O--------
                                        |        |                       |        |
             register       |0> --------O--------@-----------------------@--------O--------
                                        |        |                       |        |
                            |0> --------|--------|-----------------------|--------|--------
                                        |        |                       |        |
                            |0> --------|--------|-----------------------|--------|--------
                                        |        |                       |        |
                                        |        |                       |        |
                            |0> -------[X]-------|---------@------------[X]-------|--------
                                                 |         |                      |
        threshold register  |0> ----------------[X]--------|-----@---------------[X]-------
                                                           |     |
                            |0> --------------------------[X]---[X]------------------------

        The final qubit in the threshold_register is a "flag qubit" which determines whether or not to accept the
        recommendation. That is, recommendation results are post-selected on this qubit.

        Returns:
            None

        Modifies:
            The input circuit.
        """
        # Determine the number of controls
        ncontrols = (ctrl_string + "1").index("1")

        # Edge case in which no operations are added
        if ncontrols == 0:
            return

        # Make sure there is at least one control
        if ncontrols == len(ctrl_string):
            raise ValueError("Argument ctrl_string should have at least one '1'.")

        # Add the ancilla register of three qubits to the circuit
        ancilla = QuantumRegister(3, name="threshold")
        circuit.add_register(ancilla)

        # ==============================
        # Add the gates for thresholding
        # ==============================

        # Do the first "anti-controlled" Tofolli
        for ii in range(ncontrols):
            circuit.x(register[ii])
        mct(circuit, register[:ncontrols], ancilla[0], None, mode="noancilla")
        for ii in range(ncontrols):
            circuit.x(register[ii])

        # Do the second Tofolli
        mct(circuit, register[:ncontrols], ancilla[1], None, mode="noancilla")

        # Do the copy CNOTs in the ancilla register
        circuit.cx(ancilla[0], ancilla[2])
        circuit.cx(ancilla[1], ancilla[2])

        # Uncompute the second Tofolli
        mct(circuit, register[:ncontrols], ancilla[1], None, mode="noancilla")

        # Uncompute the first "anti-controlled" Tofolli
        for ii in range(ncontrols):
            circuit.x(register[ii])
        mct(circuit, register[:ncontrols], ancilla[0], None, mode="noancilla")
        for ii in range(ncontrols):
            circuit.x(register[ii])

        # Add the "flag qubit" measurement, if desired
        if measure_flag_qubit:
            creg = ClassicalRegister(1, name="flag")
            circuit.add_register(creg)
            circuit.measure(ancilla[2], creg[0])

    def construct_circuit(self,
                          user,
                          threshold,
                          measurements=True,
                          return_registers=False,
                          logical_barriers=False,
                          swaps=True):
        """Returns the quantum circuit to recommend product(s) to a user.

        Args:
            user : numpy.ndarray
                Mathematically, a vector whose length is equal to the number of columns in the preference matrix.
                In the context of recommendation systems, this is a vector of "ratings" for "products,"
                where each vector elements corresponds to a rating for product 0, 1, ..., N. Examples:
                    user = numpy.array([1, 0, 0, 0])
                        The user likes the first product, and we have no information about the other products.

                    user = numpy.array([0.5, 0.9, 0, 0])
                        The user is neutral about the first product, and rated the second product highly.

            threshold : float in the interval [0, 1)
                Only singular vectors with singular values greater than `threshold` are kept for making recommendations.
                Note: Singular values are assumed to be normalized (via sigma / ||P||_F) to lie in the interval [0, 1).

                This is related to the low rank approximation in the preference matrix P. The larger the threshold,
                the lower the assumed rank of P. Examples:
                    threshold = 0.
                        All singular values are kept. This means we have "complete knowledge" about the user, and
                        recommendations are made solely based off their preferences. (No quantum circuit is needed.)

                    threshold = 0.5
                        All singular values greater than 0.5 are kept for recommendations.

                    threshold = 1 (invalid)
                        No singular values are kept for recommendations. This throws an error.

            measurements : bool
                Determines whether the product register is measured at the end of the circuit.

            return_registers : bool
                If True, registers in the circuit are returned in the order:
                    (1) Circuit.
                    (2) QPE register.
                    (3) User register.
                    (4) Product register.
                    (5) Classical register for product measurements (if measurements == True).

                Note: Registers can always be accessed through the return circuit. This is provided for convenience.

            logical_barriers : bool
                Determines whether to place barriers in the circuit separating subroutines.

            swaps : bool
                If True, the product register qubits are swapped to put the measurement outcome in big endian.

        Returns : qiskit.QuantumCircuit (and qiskit.QuantumRegisters, if desired)
            A quantum circuit implementing the quantum recommendation systems algorithm.
        """
        # Make sure the user is valid
        self._validate_user(user)

        # Convert the threshold value to the control string
        ctrl_string = self._threshold_to_control_string(threshold)

        # Create the QSVE circuit
        circuit, qpe_register, user_register, product_register = self._qsve.create_circuit(
            nprecision_bits=self._precision,
            logical_barriers=logical_barriers,
            load_row_norms=True,
            init_state_col=user,
            return_registers=True,
            row_name="user",
            col_name="product"
        )

        # Add the thresholding operation on the singular value (QPE) register
        self._threshold(circuit, qpe_register, ctrl_string, measure_flag_qubit=True)

        # Add the inverse QSVE circuit
        # pylint: disable=E1101
        circuit += self._qsve.create_circuit(
            nprecision_bits=self._precision,
            logical_barriers=logical_barriers,
            load_row_norms=False,
            init_state_col=None,
            return_registers=False,
            row_name="user",
            col_name="product"
        ).inverse()

        # Swap the qubits in the product register to put resulting bit string in big endian
        if swaps:
            for ii in range(len(product_register) // 2):
                circuit.swap(product_register[ii], product_register[-ii - 1])

        # Add measurements to the product register, if desired
        if measurements:
            creg = ClassicalRegister(len(product_register), name="recommendation")
            circuit.add_register(creg)
            circuit.measure(product_register, creg)

        if return_registers:
            if measurements:
                return circuit, qpe_register, user_register, product_register, creg
            return circuit, qpe_register, user_register, product_register
        return circuit

    def run_and_return_counts(self, user, threshold):
        """Runs the quantum circuit recommending products for the given user and returns the raw counts."""
        # TODO: Potential bug, since _threshold circuit will have a measurement in it. Maybe raise error in this case?
        #  Use statevector with subsystem option, get subsystem density matrix
        if self._quantum_instance.is_statevector:
            circuit = self.construct_circuit(user, threshold, measurements=False, logical_barriers=False)
            self._ret["circuit"] = circuit
            result = self._quantum_instance.execute(circuit)
            return result.get_statevector(circuit)
        else:
            circuit = self.construct_circuit(user, threshold, measurements=True, logical_barriers=False)
            self._ret["circuit"] = circuit
            result = self._quantum_instance.execute(circuit)
            return result.get_counts(circuit)

    def recommend(self, user, threshold, with_probabilities=True, products_as_ints=True):
        """Returns a recommendation for a specified user."""
        # Run the quantum recommendation and get the counts
        counts = self.run_and_return_counts(user, threshold)

        # Remove all outcomes with flag qubit measured as zero (assuming the flag qubit is measured)
        post_selected = []
        for bits, count in counts.items():
            # This checks if two different registers have been measured (i.e., checks if the flag qubit has been
            # measured or not).
            if " " in bits:
                product, flag = bits.split()
                if flag == "1":
                    post_selected.append((product, count))
            else:
                post_selected.append((bits, count))

        # Get the number of post-selected measurements for normalization
        new_shots = sum([x[1] for x in post_selected])

        # Format the output
        products = []
        probs = []
        for (product, count) in post_selected:
            if products_as_ints:
                product = self._binary_string_to_int(product, big_endian=True)
            products.append(product)

            if with_probabilities:
                probs.append(count / new_shots)
        if with_probabilities:
            return products, probs
        return products

    def _run(self, **kwargs):
        """Executes the quantum circuit and returns the output dictionary."""
        # Store the user vector and threshold value to return in the output dictionary
        self._ret["input"] = {"user_vector": self._user, "threshold_value": self._threshold_value}

        products, probabilities = self.recommend(self._user, self._threshold_value)
        self._ret["products"] = products
        self._ret["probabilities"] = probabilities
        return self._ret

    @staticmethod
    def classical_recommendation(preference_matrix, user, rank, quantum_format=True):
        """Returns a recommendation for a specified user via classical singular value decomposition.

        Args:
            preference_matrix : numpy.ndarray
                A matrix whose rows represent "user vectors" and columns represent products.
                Mathematically, a real-valued matrix with floating point entries.

            user : numpy.ndarray
                A vector whose length is equal to the number of columns in the preference matrix.
                See help(QuantumRecommendation.create_circuit) for more context.

            rank : int in the range (0, minimum dimension of preference matrix)
                Specifies how many singular values (ordered in decreasing order) to keep in the recommendation.

            quantum_format : bool
                Specifies format of returned value(s).

                If False, a vector of product recommendations is returned.
                    For example, vector = [1, 0, 0, 1] means the first and last products are recommended.

                If True, two arguments are returned in the order:
                    (1) list<int>
                        List of integers specifying products to recommend.
                    (2) list<float>
                        List of floats specifying the probabilities to recommend products.

                    For example:
                        products = [0, 2], probabilities = [0.36, 0.64] means
                        product 0 is recommended with probability 0.36, product 2 is recommend with probability 0.64.

        Returns : Union[numpy.ndarray, tuple(list<int>, list<float>)
            See `quantum_format` above for more information.
        """
        # Do the classical SVD
        _, _, vmat = np.linalg.svd(preference_matrix, full_matrices=True)

        # Do the projection
        recommendation = np.zeros_like(user, dtype=np.float64)
        for ii in range(rank):
            recommendation += np.dot(np.conj(vmat[ii]), user) * vmat[ii]

        if np.allclose(recommendation, np.zeros_like(recommendation)):
            raise RankError("Given rank is smaller than the rank of the preference matrix. Recommendations "
                            "cannot be made for all users.")

        # Return the squared values for probabilities
        probabilities = (recommendation / np.linalg.norm(recommendation, ord=2))**2

        # Return the vector if quantum_format is False
        if not quantum_format:
            return probabilities

        # Format the same as the quantum recommendation
        prods = []
        probs = []
        for (ii, p) in enumerate(probabilities):
            if p > 0:
                prods.append(ii)
                probs.append(p)
        return prods, probs

    def _validate_user(self, user):
        """Validates a user (vector).

        If an invalid user for the recommendation system is given, a UserError is thrown. Else, nothing happens.

        Args:
            user : numpy.ndarray
                A vector representing a user in a recommendation system.
        """
        # Make sure the user is of the correct type
        if not isinstance(user, (list, tuple, np.ndarray)):
            raise UserVectorError("Invalid type for user. Accepted types are list, tuple, and numpy.ndarray.")

        # Make sure the user vector has the correct length
        if len(user) != self.num_products:
            raise UserVectorError(f"User vector should have length {self.num_products} but has length {len(user)}")

        # Make sure at least one element of the user vector is nonzero
        all_zeros = np.zeros_like(user)
        if np.allclose(user, all_zeros):
            raise UserVectorError(
                "User vector is all zero and thus contains no information. "
                "At least one element of the user vector must be nonzero."
            )

    def _validate_rank(self, rank):
        """Throws a RankError if the rank is not valid, else nothing happens."""
        if rank <= 0 or rank > self.num_users:
            raise RankError("Rank must be in the range 0 < rank <= number of users.")

    @staticmethod
    def _to_binary_decimal(decimal, nbits=5):
        """Converts a decimal in base ten to a binary decimal string.

        Args:
            decimal : float
                Floating point value in the interval [0, 1).

            nbits : int
                Number of bits to use in the binary decimal.

        Return type:
            str
        """
        return decimal_to_binary(decimal, max_num_digits=nbits)

    @staticmethod
    def _binary_string_to_int(bitstring, big_endian=True):
        """Returns the integer equivalent of the binary string.

        Args:
            bitstring : str
                String of characters "1" and "0".

            big_endian : bool
                Dictates whether string is big endian (most significant bit first) or little endian.
        """
        if not big_endian:
            bitstring = str(reversed(bitstring))
        val = 0
        nbits = len(bitstring)
        for (n, bit) in enumerate(bitstring):
            if bit == "1":
                val += 2**(nbits - n - 1)
        return val

    @staticmethod
    def _rank_to_ctrl_string(rank, length):
        """Converts the rank to a ctrl_string for thresholding.

        Args:
            rank : int
                Assumed rank of a system (cutoff for singular values).

            length : int
                Number of characters for the ctrl_string.

        Returns : str
            Binary string (base 2) representation of the rank with the given length.
        """
        raise NotImplementedError("This function is not yet implemented.")

    def _threshold_to_control_string(self, threshold):
        """Returns a control string for the threshold circuit which keeps all values strictly above the threshold."""
        # Make sure the threshold is ok
        if threshold < 0 or threshold > 1:
            raise ThresholdError("Argument threshold must satisfy 0 <= threshold <= 1.")

        # Compute the angle 0 <= theta <= 1 for this threshold singular value
        theta = 1 / np.pi * np.arccos(threshold)

        # Return the binary decimal of theta
        return self._qsve.to_binary_decimal(theta, nbits=self._precision)

    def __str__(self):
        return "Quantum Recommendation System with {} users and {} products.".format(self.num_users, self.num_products)
