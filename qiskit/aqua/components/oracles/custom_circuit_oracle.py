# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Custom Circuit-based Quantum Oracle.
"""

from typing import Optional, Callable, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua import AquaError
from .oracle import Oracle


class CustomCircuitOracle(Oracle):
    """
    The Custom Circuit-based Quantum Oracle.

    A helper class to, in essence, 'wrap' a user-supplied quantum circuit such that it becomes
    of type :class:`Oracle` and hence can be used by algorithms taking an oracle as input.

    This class is provided for easy creation of oracles using custom circuits.
    It is geared towards programmatically experimenting with oracles, where a user directly
    provides a `QuantumCircuit` object, corresponding to the intended oracle function,
    together with the various `QuantumRegister` objects involved.

    Note:
        The `evaluate_classically_callback` param is to supply a method to classically evaluate
        the function (as encoded by the oracle) on a particular input bitstring. For example
        for an oracle that encodes 3-SAT problems, this method would determine classically if
        an input variable assignment would satisfy the 3-SAT expression.

        The input bitstring is a string of 1's and 0's corresponding to the input variable(s). The
        return should be a (bool, List[int]) tuple where the bool corresponds to the return value
        of the *binary* function encoded by the oracle, and the List[int] should just be a
        different representation of the input variable assignment, which should be equivalent to
        the bitstring or a quantum measurement.

        Examples of existing implementations, for reference, can be found in other oracles such as
        :meth:`TruthTableOracle.evaluate_classically` and
        :meth:`LogicalExpressionOracle.evaluate_classically`.
    """

    def __init__(self, variable_register: QuantumRegister,
                 output_register: QuantumRegister,
                 circuit: QuantumCircuit,
                 ancillary_register: Optional[QuantumRegister] = None,
                 evaluate_classically_callback:
                 Optional[Callable[[str], Tuple[bool, List[int]]]] = None):
        """
        Args:
            variable_register: The register holding variable qubit(s) for the oracle function
            output_register: The register holding output qubit(s) for the oracle function
            circuit: The quantum circuit corresponding to the intended oracle function
            ancillary_register: The register holding ancillary qubit(s)
            evaluate_classically_callback: The classical callback function for evaluating the
                oracle, for example, to use with :class:`~qiskit.aqua.algorithms.Grover`'s search
        Raises:
            AquaError: Invalid input
        """

        super().__init__()
        if variable_register is None:
            raise AquaError('Missing QuantumRegister for variables.')
        if output_register is None:
            raise AquaError('Missing QuantumRegister for output.')
        if circuit is None:
            raise AquaError('Missing custom QuantumCircuit for the oracle.')
        self._variable_register = variable_register
        self._output_register = output_register
        self._circuit = circuit
        self._ancillary_register = ancillary_register
        if evaluate_classically_callback is not None:
            self.evaluate_classically = evaluate_classically_callback

    @property
    def variable_register(self):
        return self._variable_register

    @property
    def output_register(self):
        return self._output_register

    @property
    def ancillary_register(self):
        return self._ancillary_register

    @property
    def circuit(self):
        return self._circuit

    def construct_circuit(self):
        """Construct the oracle circuit.

        Returns:
            QuantumCircuit: A quantum circuit for the oracle.
        """
        return self._circuit
