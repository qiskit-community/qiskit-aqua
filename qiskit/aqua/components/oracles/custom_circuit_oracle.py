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
    """

    def __init__(self, variable_register: Optional[QuantumRegister] = None,
                 output_register: Optional[QuantumRegister] = None,
                 ancillary_register: Optional[QuantumRegister] = None,
                 circuit: Optional[QuantumCircuit] = None,
                 evaluate_classically_callback:
                 Optional[Callable[[str], Tuple[bool, List[int]]]] = None):
        """
        Args:
            variable_register: The register holding variable qubit(s) for
                    the oracle function
            output_register: The register holding output qubit(s)
                    for the oracle function
            ancillary_register: The register holding ancillary qubit(s)
            circuit: The quantum circuit corresponding to the
                    intended oracle function
            evaluate_classically_callback: The classical callback function for
                    evaluating the oracle, for example, to use with
                    :class:`~qiskit.aqua.algorithms.Grover`'s search
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
        self._ancillary_register = ancillary_register
        self._circuit = circuit
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
