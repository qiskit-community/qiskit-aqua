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
from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_oracle_instance

logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
    """The Grover Quantum algorithm."""

    PROP_INCREMENTAL = 'incremental'
    PROP_NUM_ITERATIONS = 'num_iterations'

    GROVER_CONFIGURATION = {
        'name': 'Grover',
        'description': 'Grover',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'grover_schema',
            'type': 'object',
            'properties': {
                PROP_INCREMENTAL: {
                    'type': 'boolean',
                    'default': False
                },
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
        self._incremental = False
        self._num_iterations = 1
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
        incremental = grover_params.get(Grover.PROP_INCREMENTAL)
        num_iterations = grover_params.get(Grover.PROP_NUM_ITERATIONS)

        oracle_params = params.get(QuantumAlgorithm.SECTION_KEY_ORACLE)
        oracle = get_oracle_instance(oracle_params['name'])
        oracle.init_params(oracle_params)
        self.init_args(oracle, incremental=incremental, num_iterations=num_iterations)

    def init_args(self, oracle, incremental=False, num_iterations=1):
        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))
        self._oracle = oracle
        self._max_num_iterations = 2 ** (len(self._oracle.variable_register()) / 2)
        self._incremental = incremental
        self._num_iterations = num_iterations
        if incremental:
            logger.debug('Incremental mode specified, ignoring "num_iterations".')
        else:
            if num_iterations > self._max_num_iterations:
                logger.warning('The specified value {} for "num_iterations" might be too high.'.format(num_iterations))

    def _construct_circuit_components(self):
        measurement_cr = ClassicalRegister(len(self._oracle.variable_register()), name='m')
        if self._oracle.ancillary_register():
            qc_prefix = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register(),
                measurement_cr
            )
            qc_amplitude_amplification = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register()
            )
        else:
            qc_prefix = QuantumCircuit(
                self._oracle.variable_register(),
                measurement_cr
            )
            qc_amplitude_amplification = QuantumCircuit(
                self._oracle.variable_register()
            )
        qc_prefix.h(self._oracle.variable_register())

        qc_amplitude_amplification += self._oracle.construct_circuit()
        qc_amplitude_amplification.h(self._oracle.variable_register())
        qc_amplitude_amplification.x(self._oracle.variable_register())
        qc_amplitude_amplification.x(self._oracle.outcome_register())
        qc_amplitude_amplification.h(self._oracle.outcome_register())
        if self._oracle.ancillary_register():
            qc_amplitude_amplification.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                [self._oracle.ancillary_register()[i] for i in range(len(self._oracle.ancillary_register()))],
                self._oracle.outcome_register()[0]
            )
        else:
            qc_amplitude_amplification.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                [],
                self._oracle.outcome_register()[0]
            )
        qc_amplitude_amplification.h(self._oracle.outcome_register())
        qc_amplitude_amplification.x(self._oracle.variable_register())
        qc_amplitude_amplification.x(self._oracle.outcome_register())
        qc_amplitude_amplification.h(self._oracle.variable_register())
        qc_amplitude_amplification.h(self._oracle.outcome_register())

        qc_measurement = QuantumCircuit(
            self._oracle.variable_register(),
            measurement_cr
        )
        qc_measurement.measure(self._oracle.variable_register(), measurement_cr)

        return qc_prefix, qc_amplitude_amplification, qc_measurement

    def _run_with_num_iterations(self, qc_prefix, qc_amplitude_amplification, qc_measurement):
        qc = qc_prefix + qc_amplitude_amplification + qc_measurement
        self._ret['circuit'] = qc
        self._ret['measurements'] = self.execute(qc).get_counts(qc)
        assignment = self._oracle.interpret_measurement(self._ret['measurements'])
        oracle_evaluation = self._oracle.evaluate_classically(assignment)
        return assignment, oracle_evaluation

    def run(self):
        qc_prefix, qc_amplitude_amplification, qc_measurement = self._construct_circuit_components()

        if self._incremental:
            qc_amplitude_amplification_single_iteration_data = qc_amplitude_amplification.data
            current_num_iterations = 1
            while current_num_iterations <= self._max_num_iterations:
                assignment, oracle_evaluation = self._run_with_num_iterations(
                    qc_prefix, qc_amplitude_amplification, qc_measurement
                )
                if oracle_evaluation:
                    break
                current_num_iterations += 1
                qc_amplitude_amplification.data += qc_amplitude_amplification_single_iteration_data
        else:
            qc_amplitude_amplification.data *= self._num_iterations
            assignment, oracle_evaluation = self._run_with_num_iterations(
                qc_prefix, qc_amplitude_amplification, qc_measurement
            )

        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation
        return self._ret
