# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
"""
The Custom Circuit-based Quantum Oracle.
"""

from qiskit.aqua import Pluggable, AquaError

from .oracle import Oracle


class CustomCircuitOracle(Oracle):
    CONFIGURATION = {
        'name': 'CustomCircuitOracle',
        'description': 'Custom Circuit Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'custom_circuit_oracle_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, variable_register=None, output_register=None, ancillary_register=None, circuit=None):
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

    @classmethod
    def init_params(cls, params):
        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        args = {k: v for k, v in oracle_params.items() if k != 'name'}
        return cls(**args)

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
            A quantum circuit for the oracle.
        """
        raise self._circuit
