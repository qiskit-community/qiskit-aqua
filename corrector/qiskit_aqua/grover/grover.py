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
import warnings

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_oracle_instance

logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
    """The Grover Quantum algorithm."""

    PROP_NUM_ITERATIONS = 'num_iterations'

    GROVER_CONFIGURATION = {
        'name': 'Grover',
        'description': 'Grover',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'grover_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_ITERATIONS: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
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
        self._num_iterations = None
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

        grover_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_iterations = grover_params.get(Grover.PROP_NUM_ITERATIONS)

        oracle_params = params.get(QuantumAlgorithm.SECTION_KEY_ORACLE)
        oracle = get_oracle_instance(oracle_params['name'])
        oracle.init_params(oracle_params)
        self.init_args(oracle, num_iterations=num_iterations)

    def init_args(self, oracle, num_iterations=1):
        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))
        self._oracle = oracle
        self._num_iterations = num_iterations

    def _construct_circuit(self):
        measurement_cr = ClassicalRegister(len(self._oracle.variable_register()), name='m')
        if self._oracle.ancillary_register():
            qc = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register(),
                measurement_cr
            )
            qc_single_iteration = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register()
            )
        else:
            qc = QuantumCircuit(
                self._oracle.variable_register(),
                measurement_cr
            )
            qc_single_iteration = QuantumCircuit(
                self._oracle.variable_register()
            )
        qc.h(self._oracle.variable_register())
        qc_single_iteration += self._oracle.construct_circuit()
        qc_single_iteration.h(self._oracle.variable_register())
        qc_single_iteration.x(self._oracle.variable_register())
        qc_single_iteration.x(self._oracle.outcome_register())
        qc_single_iteration.h(self._oracle.outcome_register())
        if self._oracle.ancillary_register():
            qc_single_iteration.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                [self._oracle.ancillary_register()[i] for i in range(len(self._oracle.ancillary_register()))],
                self._oracle.outcome_register()[0]
            )
        else:
            qc_single_iteration.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                [],
                self._oracle.outcome_register()[0]
            )
        qc_single_iteration.h(self._oracle.outcome_register())
        qc_single_iteration.x(self._oracle.variable_register())
        qc_single_iteration.x(self._oracle.outcome_register())
        qc_single_iteration.h(self._oracle.variable_register())
        qc_single_iteration.h(self._oracle.outcome_register())

        qc_single_iteration.data *= self._num_iterations
        qc += qc_single_iteration

        qc.measure(self._oracle.variable_register(), measurement_cr)
        return qc

    def run(self):
        qc = self._construct_circuit()
        self._ret['circuit'] = qc
        self._ret['measurements'] = self.execute(qc).get_counts(qc)
        self._ret['result'] = self._oracle.interpret_measurement(self._ret['measurements'])
        return self._ret
