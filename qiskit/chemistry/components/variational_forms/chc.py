# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Compact heuristic ansatz for Chemistry """

from typing import List, Optional, Union

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.circuit import ParameterVector, Parameter

from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states import InitialState


class CHC(VariationalForm):
    """ This trial wavefunction is the Compact Heuristic for Chemistry.

    The trial wavefunction is as defined in
    Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855. It aims at approximating
    the UCC Ansatz for a lower CNOT count.

    Note:
        It is not particle number conserving and the accuracy of the approximation decreases
        with the number of excitations.
    """

    def __init__(self, num_qubits: int, depth: int = 1, ladder: bool = False,
                 excitations: Optional[List[List[int]]] = None,
                 entangler_map: Optional[List[int]] = None,
                 entanglement: str = 'full',
                 initial_state: Optional[InitialState] = None) -> None:
        """

        Args:
            num_qubits:
            depth:
            ladder:
            excitations:
            entangler_map:
            entanglement:
            initial_state:
        """

        super().__init__()
        self._num_qubits = num_qubits
        self._depth = depth
        self._excitations = None
        self._entangler_map = None
        self._initial_state = None
        self._ladder = ladder
        self._num_parameters = len(excitations)*depth
        self._excitations = excitations
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters
        self._num_qubits = num_qubits
        if entangler_map is None:
            self._entangler_map = VariationalForm.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = VariationalForm.validate_entangler_map(entangler_map, num_qubits)
        self._initial_state = initial_state
        self._support_parameterized_circuit = True

    def construct_circuit(self, parameters: Union[np.ndarray, List[Parameter], ParameterVector],
                              q: Optional[QuantumRegister] = None) -> QuantumCircuit:
        """
        Construct the variational form, given its parameters.

        Args:
            parameters: circuit parameters
            q: Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
            ValueError: only supports single and double excitations at the moment.
        """

        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        count = 0
        for _ in range(self._depth):
            for idx in self._excitations:

                if len(idx) == 2:

                    i = idx[0]
                    r = idx[1]

                    circuit.u1(-parameters[count] / 4 + np.pi / 4, q[i])
                    circuit.u1(-parameters[count] / 4 - np.pi / 4, q[r])

                    circuit.h(q[i])
                    circuit.h(q[r])

                    if self._ladder:
                        for qubit in range(i, r):
                            circuit.cx(q[qubit], q[qubit + 1])
                    else:
                        circuit.cx(q[i], q[r])

                    circuit.u1(parameters[count], q[r])

                    if self._ladder:
                        for qubit in range(r, i, -1):
                            circuit.cx(q[qubit - 1], q[qubit])
                    else:
                        circuit.cx(q[i], q[r])

                    circuit.h(q[i])
                    circuit.h(q[r])

                    circuit.u1(-parameters[count] / 4 - np.pi / 4, q[i])
                    circuit.u1(-parameters[count] / 4 + np.pi / 4, q[r])

                elif len(idx) == 4:

                    i = idx[0]
                    r = idx[1]
                    j = idx[2]
                    s = idx[3]

                    circuit.u1(-np.pi / 2, q[r])

                    circuit.h(q[i])
                    circuit.h(q[r])
                    circuit.h(q[j])
                    circuit.h(q[s])

                    if self._ladder:
                        for qubit in range(i, r):
                            circuit.cx(q[qubit], q[qubit+1])
                            circuit.barrier(q[qubit], q[qubit+1])
                    else:
                        circuit.cx(q[i], q[r])
                    circuit.cx(q[r], q[j])
                    if self._ladder:
                        for qubit in range(j, s):
                            circuit.cx(q[qubit], q[qubit+1])
                            circuit.barrier(q[qubit], q[qubit + 1])
                    else:
                        circuit.cx(q[j], q[s])

                    circuit.u1(parameters[count], q[s])

                    if self._ladder:
                        for qubit in range(s, j, -1):
                            circuit.cx(q[qubit-1], q[qubit])
                            circuit.barrier(q[qubit-1], q[qubit])
                    else:
                        circuit.cx(q[j], q[s])
                    circuit.cx(q[r], q[j])
                    if self._ladder:
                        for qubit in range(r, i, -1):
                            circuit.cx(q[qubit-1], q[qubit])
                            circuit.barrier(q[qubit - 1], q[qubit])
                    else:
                        circuit.cx(q[i], q[r])

                    circuit.h(q[i])
                    circuit.h(q[r])
                    circuit.h(q[j])
                    circuit.h(q[s])

                    circuit.u1(-parameters[count] / 2 + np.pi / 2, q[i])
                    circuit.u1(-parameters[count] / 2 + np.pi, q[r])

                else:
                    raise ValueError('Limited to single and double excitations, '
                                     'higher order is not implemented')

                count += 1

        return circuit
