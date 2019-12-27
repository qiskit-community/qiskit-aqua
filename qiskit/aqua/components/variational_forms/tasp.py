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


"""TASP Ansatz described in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.92.042303"""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.utils.validation import validate
from qiskit.aqua.components.variational_forms import VariationalForm


class TASP(VariationalForm):
    """Trotterized Adiabatic State Preparation.
    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.92.042303"""

    _INPUT_SCHEMA = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'id': 'tasp_schema',
        'type': 'object',
        'properties': {
            'depth': {
                'type': 'integer',
                'default': 1,
                'minimum': 1
            },
        },
        'additionalProperties': False
    }

    def __init__(self, num_qubits, h_list, depth=1, initial_state=None):
        """Constructor.

        Args:
            num_qubits (int) : number of qubits
            depth (int) : number of TASP steps layers (corresponds to the variable S in
                          equation 8 of the paper above)
            h_list (WeightedPauliOperator) : list of Hamiltonians with which to evolve,
                                             e.g. H_ex, H_hop, H_diag in the paper above.
            initial_state (InitialState): an initial state object
        """

        validate(locals(), self._INPUT_SCHEMA)
        super().__init__()
        self._num_qubits = num_qubits
        self._h_list = h_list
        self._depth = depth
        self._initial_state = initial_state
        self._num_parameters = len(self._h_list)*depth
        self._bounds = [(-np.pi, np.pi)] * self._num_parameters

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray): circuit parameters
            q (QuantumRegister): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        for _ in range(self._depth):
            for i in range(len(self._h_list)):
                if not self._h_list[i].is_empty():
                    circuit += self._h_list[i].evolve(evo_time=parameters[i], quantum_registers=q)

            for i in range(len(self._h_list)-1, -1, -1):
                if not self._h_list[i].is_empty():
                    circuit += self._h_list[i].evolve(evo_time=parameters[i], quantum_registers=q)
        return circuit
