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
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.qasm import pi
from qiskit_aqua.algorithms import QuantumAlgorithm
from qiskit_aqua import AquaError, PluggableType, get_pluggable_class

logger = logging.getLogger(__name__)


class Grover(QuantumAlgorithm):
    """
    The Grover Quantum algorithm.

    If the `num_iterations` param is specified, the amplitude amplification iteration will be built as specified.

    If the `incremental` mode is specified, which indicates that the optimal `num_iterations` isn't known in advance,
    a multi-round schedule will be followed with incremental trial `num_iterations` values.
    The implementation follows Section 4 of Boyer et al. <https://arxiv.org/abs/quant-ph/9605034>
    """

    PROP_INCREMENTAL = 'incremental'
    PROP_NUM_ITERATIONS = 'num_iterations'
    PROP_CNX_MODE = 'cnx_mode'

    CONFIGURATION = {
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
                },
                PROP_CNX_MODE: {
                    'type': 'string',
                    'default': 'basic',
                    'oneOf': [
                        {'enum': [
                            'basic',
                            'advanced'
                        ]}
                    ]
                },

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

    def __init__(self, oracle, incremental=False, num_iterations=1, cnx_mode='basic'):
        """
        Constructor.

        Args:
            oracle (Oracle): the oracle pluggable component
            incremental (bool): boolean flag for whether to use incremental search mode or not
            num_iterations (int): the number of iterations to use for amplitude amplification
        """
        self.validate(locals())
        super().__init__()
        self._oracle = oracle
        self._max_num_iterations = 2 ** (len(self._oracle.variable_register()) / 2)
        self._incremental = incremental
        self._num_iterations = num_iterations if not incremental else 1
        self._cnx_mode = cnx_mode
        self.validate(locals())
        if incremental:
            logger.debug('Incremental mode specified, ignoring "num_iterations".')
        else:
            if num_iterations > self._max_num_iterations:
                logger.warning('The specified value {} for "num_iterations" might be too high.'.format(num_iterations))
        self._ret = {}
        self._qc_prefix = None
        self._qc_amplitude_amplification_single_iteration = None
        self._qc_amplitude_amplification = None
        self._qc_measurement = None

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: input instance
        """
        if algo_input is not None:
            raise AquaError("Unexpected Input instance.")

        grover_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        incremental = grover_params.get(Grover.PROP_INCREMENTAL)
        num_iterations = grover_params.get(Grover.PROP_NUM_ITERATIONS)
        cnx_mode = grover_params.get(Grover.PROP_CNX_MODE)

        oracle_params = params.get(QuantumAlgorithm.SECTION_KEY_ORACLE)
        oracle = get_pluggable_class(PluggableType.ORACLE,
                                     oracle_params['name']).init_params(oracle_params)
        return cls(oracle, incremental=incremental, num_iterations=num_iterations, cnx_mode=cnx_mode)

    def _construct_circuit_components(self):
        measurement_cr = ClassicalRegister(len(self._oracle.variable_register()), name='m')
        if self._oracle.ancillary_register():
            qc_prefix = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register(),
                measurement_cr
            )
            qc_amplitude_amplification_single_iteration = QuantumCircuit(
                self._oracle.variable_register(),
                self._oracle.ancillary_register()
            )
        else:
            qc_prefix = QuantumCircuit(
                self._oracle.variable_register(),
                measurement_cr
            )
            qc_amplitude_amplification_single_iteration = QuantumCircuit(
                self._oracle.variable_register()
            )
        qc_prefix.u2(0, pi, self._oracle.variable_register())  # h

        qc_amplitude_amplification_single_iteration += self._oracle.construct_circuit()
        qc_amplitude_amplification_single_iteration.u2(0, pi, self._oracle.variable_register())  # h
        qc_amplitude_amplification_single_iteration.u3(pi, 0, pi, self._oracle.variable_register())  # x
        qc_amplitude_amplification_single_iteration.u3(pi, 0, pi, self._oracle.outcome_register())  # x
        qc_amplitude_amplification_single_iteration.u2(0, pi, self._oracle.outcome_register())  # h
        if self._oracle.ancillary_register():
            qc_amplitude_amplification_single_iteration.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                self._oracle.outcome_register()[0],
                [self._oracle.ancillary_register()[i] for i in range(len(self._oracle.ancillary_register()))],
                mode=self._cnx_mode
            )
        else:
            qc_amplitude_amplification_single_iteration.cnx(
                [self._oracle.variable_register()[i] for i in range(len(self._oracle.variable_register()))],
                self._oracle.outcome_register()[0],
                [],
                mode=self._cnx_mode
            )
        qc_amplitude_amplification_single_iteration.u2(0, pi, self._oracle.outcome_register())  # h
        qc_amplitude_amplification_single_iteration.u3(pi, 0, pi, self._oracle.variable_register())  # x
        qc_amplitude_amplification_single_iteration.u3(pi, 0, pi, self._oracle.outcome_register())  # x
        qc_amplitude_amplification_single_iteration.u2(0, pi, self._oracle.variable_register())  # h
        qc_amplitude_amplification_single_iteration.u2(0, pi, self._oracle.outcome_register())  # h

        qc_measurement = QuantumCircuit(
            self._oracle.variable_register(),
            measurement_cr
        )
        qc_measurement.barrier(self._oracle.variable_register())
        qc_measurement.measure(self._oracle.variable_register(), measurement_cr)

        self._qc_prefix = qc_prefix
        self._qc_measurement = qc_measurement
        self._qc_amplitude_amplification_single_iteration = qc_amplitude_amplification_single_iteration

    def _run_with_num_iterations(self):
        qc = self.construct_circuit()
        self._ret['circuit'] = qc
        self._ret['measurements'] = self._quantum_instance.execute(qc).get_counts(qc)
        assignment = self._oracle.interpret_measurement(self._ret['measurements'])
        oracle_evaluation = self._oracle.evaluate_classically(assignment)
        return assignment, oracle_evaluation

    def construct_circuit(self):
        """
        Construct the quantum circuit

        Returns:
            the QuantumCircuit object for the constructed circuit
        """
        if self._qc_prefix is None or self._qc_amplitude_amplification_single_iteration is None or self._qc_measurement is None:
            self._construct_circuit_components()
        if self._qc_amplitude_amplification is None:
            self._qc_amplitude_amplification = QuantumCircuit() + self._qc_amplitude_amplification_single_iteration
        qc = self._qc_prefix + self._qc_amplitude_amplification + self._qc_measurement
        return qc

    def _run(self):

        if self._quantum_instance.is_statevector:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(
                self._quantum_instance.backend_name))

        if self._qc_prefix is None or self._qc_amplitude_amplification_single_iteration is None or self._qc_measurement is None:
            self._construct_circuit_components()

        if self._incremental:
            current_max_num_iterations, lam = 1, 6 / 5

            def _try_current_max_num_iterations():
                target_num_iterations = np.random.randint(current_max_num_iterations) + 1
                self._qc_amplitude_amplification = QuantumCircuit()
                for _ in range(target_num_iterations):
                    self._qc_amplitude_amplification += self._qc_amplitude_amplification_single_iteration
                return self._run_with_num_iterations()

            while current_max_num_iterations < self._max_num_iterations:
                assignment, oracle_evaluation = _try_current_max_num_iterations()
                if oracle_evaluation:
                    break
                current_max_num_iterations = min(lam * current_max_num_iterations, self._max_num_iterations)
            else:
                # after max is reached, try one more round
                assignment, oracle_evaluation = _try_current_max_num_iterations()
        else:
            self._qc_amplitude_amplification = QuantumCircuit()
            for i in range(self._num_iterations):
                self._qc_amplitude_amplification += self._qc_amplitude_amplification_single_iteration
            assignment, oracle_evaluation = self._run_with_num_iterations()

        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation
        return self._ret
