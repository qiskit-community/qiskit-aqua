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
"""
The Grover Quantum algorithm.
"""

import logging

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_acqua import QuantumAlgorithm, AlgorithmError
from qiskit_acqua import get_oracle_instance


logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
    """The Grover Quantum algorithm."""
    GROVER_CONFIGURATION = {
        'name': 'Grover',
        'description': 'Grover',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'grover_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['search'],
        'depends': ['oracle'],
        'defaults': {
            'oracle': {
                'name': 'SAT'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.GROVER_CONFIGURATION.copy())
        self._oracle = None
        self._ret = {}

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: input instance
        """
        if algo_input is not None:
            raise AlgorithmError("Unexpected Input instance.")

        oracle_params = params.get(QuantumAlgorithm.SECTION_KEY_ORACLE)
        oracle = get_oracle_instance(oracle_params['name'])
        oracle.init_params(oracle_params)
        self.init_args(oracle)

    def init_args(self, oracle):
        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))
        self._oracle = oracle

    def _construct_circuit(self):
        measurement_cr = ClassicalRegister(len(self._oracle.variable_register()), name='m')
        qc = QuantumCircuit(
            self._oracle.variable_register(),
            self._oracle.ancillary_register(),
            measurement_cr
        )
        qc.h(self._oracle.variable_register())
        qc.extend(self._oracle.construct_circuit())
        qc.h(self._oracle.variable_register())
        qc.x(self._oracle.variable_register())
        qc.x(self._oracle.target_register())
        qc.h(self._oracle.target_register())
        qc.cnx(
            [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
            [self._oracle.ancillary_register()[i] for i in range(len(self._oracle.ancillary_register()))],
            self._oracle.target_register()[0]
        )
        qc.h(self._oracle.target_register())
        qc.x(self._oracle.variable_register())
        qc.x(self._oracle.target_register())
        qc.h(self._oracle.variable_register())
        qc.h(self._oracle.target_register())

        qc.measure(self._oracle.variable_register(), measurement_cr)
        return qc

    def run(self):
        qc = self._construct_circuit()
        self._ret['circuit'] = qc
        self._ret['measurements'] = self.execute(qc).get_counts(qc)
        self._ret['result'] = self._oracle.interpret_measurement(self._ret['measurements'])
        return self._ret
