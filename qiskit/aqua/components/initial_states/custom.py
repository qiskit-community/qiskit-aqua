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

import numpy as np
import logging

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import execute as q_execute
from qiskit import BasicAer

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.utils.arithmetic import normalize_vector
from qiskit.aqua.utils.circuit_utils import convert_to_basis_gates

logger = logging.getLogger(__name__)


class Custom(InitialState):
    """A custom initial state."""

    CONFIGURATION = {
        'name': 'CUSTOM',
        'description': 'Custom initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'custom_state_schema',
            'type': 'object',
            'properties': {
                'state': {
                    'type': 'string',
                    'default': 'zero',
                    'oneOf': [
                        {'enum': ['zero', 'uniform', 'random']}
                    ]
                },
                'state_vector': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, state="zero", state_vector=None, circuit=None):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            state (str): `zero`, `uniform` or `random`
            state_vector: customized vector
            circuit (QuantumCircuit): the actual custom circuit for the desired initial state
        """
        loc = locals().copy()
        # since state_vector is a numpy array of complex numbers which aren't json valid,
        # remove it from validation
        del loc['state_vector']
        self.validate(loc)
        super().__init__()
        self._num_qubits = num_qubits
        self._state = state
        size = np.power(2, self._num_qubits)
        self._circuit = None
        if circuit is not None:
            if circuit.width() != num_qubits:
                logger.warning('The specified num_qubits and the provided custom circuit do not match.')
            self._circuit = convert_to_basis_gates(circuit)
            if state_vector is not None:
                self._state = None
                self._state_vector = None
                logger.warning('The provided state_vector is ignored in favor of the provided custom circuit.')
        else:
            if state_vector is None:
                if self._state == 'zero':
                    self._state_vector = np.array([1.0] + [0.0] * (size - 1))
                elif self._state == 'uniform':
                    self._state_vector = np.array([1.0 / np.sqrt(size)] * size)
                elif self._state == 'random':
                    self._state_vector = normalize_vector(aqua_globals.random.rand(size))
                else:
                    raise AquaError('Unknown state {}'.format(self._state))
            else:
                if len(state_vector) != np.power(2, self._num_qubits):
                    raise AquaError('The state vector length {} is incompatible with the number of qubits {}'.format(
                        len(state_vector), self._num_qubits
                    ))
                self._state_vector = normalize_vector(state_vector)
                self._state = None

    def construct_circuit(self, mode, register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): register for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            AquaError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            if self._state_vector is None:
                if self._circuit is not None:
                    self._state_vector = np.asarray(q_execute(self._circuit, BasicAer.get_backend(
                        'statevector_simulator')).result().get_statevector(self._circuit))
            return self._state_vector
        elif mode == 'circuit':
            if self._circuit is None:
                if register is None:
                    register = QuantumRegister(self._num_qubits, name='q')

                # create emtpy quantum circuit
                circuit = QuantumCircuit()

                # if register is actually a list of qubits
                if type(register) is list:

                    # loop over all qubits and add the required registers
                    for q in register:
                        if not circuit.has_register(q[0]):
                            circuit.add_register(q[0])
                else:
                    # if an actual register is given, add it
                    circuit.add_register(register)

                if self._state is None or self._state == 'random':
                    svc = StateVectorCircuit(self._state_vector)
                    svc.construct_circuit(circuit, register)
                elif self._state == 'zero':
                    pass
                elif self._state == 'uniform':
                    for i in range(self._num_qubits):
                        circuit.u2(0.0, np.pi, register[i])
                else:
                    pass
                self._circuit = circuit
            return self._circuit.copy()
        else:
            raise AquaError('Mode should be either "vector" or "circuit"')
