# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Oracle utilities to use with the Grover algorithm.
"""
from abc import abstractmethod

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua.circuits.gates import mct

from .custom_circuit_oracle import CustomCircuitOracle

class FunctionInversionOracle(CustomCircuitOracle):
    """
    A helper to invert a function in the form of a user-supplied circuit.

    The helper modifies the circuit in-place to build a binary function
    that evaluates to True when the output register has some specific values:
    the targets.

    This new circuit becomes the oracle and it's useful for using with the
    Grover algorithm to invert an arbitrary, possibly multi-output function at
    the target image points.

    Users of this class must subclass it and implement the ``evaluate()``
    method that classically executes the operation in the circuit over
    a measurement.

    Targets are passed to the constructor as a list of text strings with
    the binary representations of the values for which the boolean funcition
    will be True. A character distinct of ``'0'`` or ``'1'`` can be used to
    express indiference regarding the bit. For instance, the following lists
    represent the same target values: ``['10', '11']`` and ``['1X']``.

    Since the circuit is modified in-place, it cannot be reused. If reusing
    the circuit is needed, pass a copy of it and **ensure you're passing
    the registers of the copy**.

    Args:
        targets (list of str): The desired values for which we want to invert the function.
        variable_register (QuantumRegister): The register holding the input qubit(s) of the function.
        output_register (QuantumRegister): The register holding the output qubit(s) of the function.
        ancillary_register (QuantumRegister): The register holding ancillary qubit(s)
        circuit (QuantumCircuit): The quantum circuit corresponding to the intended oracle function
    """

    def __init__(self, targets=None, variable_register=None,
        output_register=None, ancillary_register=None, circuit=None):

        self._targets = targets
        binary_output = self._build_comparator(circuit, output_register)
        super().__init__(variable_register, binary_output, ancillary_register,
            circuit)

    def _build_comparator(self, circuit, output_register):
        inverse = circuit.inverse()
        boolean_output = self._setup_boolean_function(circuit, output_register)
        # uncompute
        circuit.extend(inverse)
        return boolean_output

    def _setup_boolean_function(self, circuit, output_register):
        boolean_output = QuantumRegister(1, 'boolean__output')
        circuit.add_register(boolean_output)

        ancilla_needed = max(map(len, self._targets)) - 2
        ancilla_register = QuantumRegister(ancilla_needed, 'mct_ancilla')
        circuit.add_register(ancilla_register)

        for one_target in self._targets:
            little_endian_target = reversed(one_target)
            control_indices = [
                (index, control_type)
                for index, control_type in enumerate(little_endian_target)
                if control_type in '10'
            ]
            self._add_mct(
                circuit, output_register,
                control_indices, boolean_output, ancilla_register)

        return boolean_output

    def _add_mct(
        self, circuit, output_register,
        control_indices, boolean_output, ancilla_register):

        # add X gates to implement controlling the Toffoli when 0
        for index, control_type in control_indices:
            if control_type == '0':
                circuit.x(output_register[index])

        circuit.mct(
            [output_register[index] for index, _ in control_indices],
            boolean_output[0], ancilla_register)

        # uncompute the X gates
        for index, control_type in control_indices:
            if control_type == '0':
                circuit.x(output_register[index])

    def evaluate_classically(self, measurement):
        """
        Check if the measurement is one of the desired targets passed to
        the constructor.

        Args:
            measurement (str): A text string representing a binary representation of the measurement.

        Returns:
            A pair with the result of the check (True or False) and the representation of the measurement.
        """
        is_ok = self.evaluate(measurement) in self._targets
        representation = self.representation(measurement)
        return is_ok, representation

    @abstractmethod
    def evaluate(self, measurement):
        """
        Perform the circuit operation classically.

        Args:
            measurement (str): A text string representing a binary representation of the measurement.
        """
        raise NotImplementedError

    def representation(self, measurement):
        """
        Args:
            measurement (str): A text string representing a binary representation of the measurement.

        Returns:
            The representation of the measurement.
        """
        return measurement
