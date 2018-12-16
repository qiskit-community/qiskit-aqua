# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import CompositeGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
import numpy as np

from qiskit_aqua.components.initial_states import InitialState


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

    def __init__(self, num_qubits, state="zero", state_vector=None):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            state (str): `zero`, `uniform` or `random`
            state_vector: customized vector
        """
        loc = locals().copy()
        del loc['state_vector']
        self.validate(loc)
        super().__init__()
        # since state_vector is a numpy array of complex numbers which aren't json valid,
        # remove it from validation
        self._num_qubits = num_qubits
        self._state = state
        size = np.power(2, self._num_qubits)
        if state_vector is None:
            if self._state == 'zero':
                self._state_vector = np.array([1.0] + [0.0] * (size - 1))
            elif self._state == 'uniform':
                self._state_vector = np.array([1.0 / np.sqrt(size)] * size)
            elif self._state == 'random':
                self._state_vector = Custom._normalize(np.random.rand(size))
            else:
                raise ValueError('Unknown state {}'.format(self._state))
        else:
            if len(state_vector) != np.power(2, self._num_qubits):
                raise ValueError('State vector length {} incompatible with num qubits {}'
                                 .format(len(state_vector), self._num_qubits))
            self._state_vector = Custom._normalize(state_vector)
            self._state = None

    @staticmethod
    def _normalize(vector):
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _convert_to_basis_gates(gates):
        if isinstance(gates, list):
            return [Custom._convert_to_basis_gates(gate) for gate in gates]
        elif isinstance(gates, CompositeGate):
            gates_data = [Custom._convert_to_basis_gates(
                gate) for gate in gates.data]
            gates = CompositeGate(gates.name, gates.param,
                                  gates.qargs, circuit=gates.circuit)
            gates.data = gates_data
            return gates
        else:
            if isinstance(gates, RYGate):
                return U3Gate(gates.param[0], 0, 0, gates.qargs[0])
            elif isinstance(gates, RZGate):
                return U1Gate(gates.param[0], gates.qargs[0])
            elif isinstance(gates, CnotGate):
                return gates
            else:
                raise RuntimeError(
                    'Unexpected component {} from the initialization circuit.'.format(gates.qasm()))

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
            ValueError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            return self._state_vector
        elif mode == 'circuit':
            if register is None:
                register = QuantumRegister(self._num_qubits, name='q')
            circuit = QuantumCircuit(register)

            if self._state is None or self._state == 'random':
                circuit.initialize(self._state_vector, [
                                   register[i] for i in range(self._num_qubits)])
                circuit.data = Custom._convert_to_basis_gates(circuit.data)
            elif self._state == 'zero':
                pass
            elif self._state == 'uniform':
                for i in range(self._num_qubits):
                    circuit.u2(0.0, np.pi, register[i])
            else:
                pass

            return circuit
        else:
            raise ValueError('Mode should be either "vector" or "circuit"')
