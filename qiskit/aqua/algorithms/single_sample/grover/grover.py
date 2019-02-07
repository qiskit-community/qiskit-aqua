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
import operator

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.qasm import pi

from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.initial_states import Custom


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
    PROP_MCT_MODE = 'mct_mode'

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
                PROP_MCT_MODE: {
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
        'depends': [
            {'pluggable_type': 'initial_state',
             'default': {
                 'name': 'CUSTOM',
                 'state': 'uniform'
             }
             },
            {'pluggable_type': 'oracle',
             'default': {
                     'name': 'SAT',
                },
             },
        ],
    }

    def __init__(self, oracle, init_state=None, incremental=False, num_iterations=1, mct_mode='basic'):
        """
        Constructor.

        Args:
            oracle (Oracle): the oracle pluggable component
            init_state (InitialState): the initial quantum state preparation
            incremental (bool): boolean flag for whether to use incremental search mode or not
            num_iterations (int): the number of iterations to use for amplitude amplification
        """
        self.validate(locals())
        super().__init__()
        self._oracle = oracle
        self._mct_mode = mct_mode
        self._oracle_circuit = oracle.construct_circuit()
        self._init_state = init_state if init_state else Custom(len(oracle.variable_register), state='uniform')
        self._init_state_circuit = self._init_state.construct_circuit(mode='circuit', register=oracle.variable_register)
        self._init_state_circuit_inverse = self._init_state_circuit.copy()
        self._init_state_circuit_inverse.data = [
            g.inverse() for g in reversed(
                self._init_state_circuit_inverse.data
            )
        ]
        self._diffusion_circuit = Grover.construct_diffusion_circuit(
            self._oracle.variable_register,
            self._oracle.ancillary_register,
            mct_mode=mct_mode
        )
        self._max_num_iterations = np.ceil(2 ** (len(oracle.variable_register) / 2))
        self._incremental = incremental
        self._num_iterations = num_iterations if not incremental else 1
        self.validate(locals())
        if incremental:
            logger.debug('Incremental mode specified, ignoring "num_iterations".')
        else:
            if num_iterations > self._max_num_iterations:
                logger.warning('The specified value {} for "num_iterations" might be too high.'.format(num_iterations))
        self._ret = {}
        self._qc_aa_iteration = None
        self._qc_amplitude_amplification = None
        self._qc_measurement = None

    @staticmethod
    def construct_diffusion_circuit(variable_register, ancillary_register, mct_mode='basic'):
        qc = QuantumCircuit(variable_register)
        if ancillary_register:
            qc.add_register(ancillary_register)
        num_variable_qubits = len(variable_register)
        qc.barrier(variable_register)
        qc.u3(pi, 0, pi, variable_register)
        qc.u2(0, pi, variable_register[num_variable_qubits - 1])
        qc.mct(
            variable_register[0:num_variable_qubits - 1],
            variable_register[num_variable_qubits - 1],
            ancillary_register,
            mode=mct_mode
        )
        qc.u2(0, pi, variable_register[num_variable_qubits - 1])
        qc.u3(pi, 0, pi, variable_register)
        qc.barrier(variable_register)
        return qc

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

        grover_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        incremental = grover_params.get(Grover.PROP_INCREMENTAL)
        num_iterations = grover_params.get(Grover.PROP_NUM_ITERATIONS)
        mct_mode = grover_params.get(Grover.PROP_MCT_MODE)

        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        oracle = get_pluggable_class(PluggableType.ORACLE,
                                     oracle_params['name']).init_params(params)

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = len(oracle.variable_register)
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(params)

        return cls(oracle, init_state=init_state,
                   incremental=incremental, num_iterations=num_iterations, mct_mode=mct_mode)

    @property
    def qc_amplitude_amplification_iteration(self):
        if self._qc_aa_iteration is None:
            self._qc_aa_iteration = QuantumCircuit()
            self._qc_aa_iteration += self._oracle_circuit
            self._qc_aa_iteration += self._init_state_circuit_inverse
            self._qc_aa_iteration += self._diffusion_circuit
            self._qc_aa_iteration += self._init_state_circuit
        return self._qc_aa_iteration

    def _run_with_existing_iterations(self):
        qc = self.construct_circuit()
        if self._quantum_instance.is_statevector:
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            variable_register_density_matrix = get_subsystem_density_matrix(
                complete_state_vec,
                range(len(self._oracle.variable_register), qc.width())
            )
            variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
            max_amplitude = max(
                variable_register_density_matrix_diag.min(),
                variable_register_density_matrix_diag.max(),
                key=abs
            )
            max_amplitude_idx = np.where(variable_register_density_matrix_diag == max_amplitude)[0][0]
            top_measurement = format(max_amplitude_idx, '0{}b'.format(len(self._oracle.variable_register)))
        else:
            measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            qc.add_register(measurement_cr)
            qc.measure(self._oracle.variable_register, measurement_cr)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement
        assignment = self._oracle.interpret_measurement(top_measurement=top_measurement)
        oracle_evaluation = self._oracle.evaluate_classically(assignment)
        return assignment, oracle_evaluation

    def construct_circuit(self):
        """
        Construct the quantum circuit

        Returns:
            the QuantumCircuit object for the constructed circuit
        """
        if self._qc_amplitude_amplification is None:
            self._qc_amplitude_amplification = QuantumCircuit() + self.qc_amplitude_amplification_iteration
        qc = QuantumCircuit(self._oracle.variable_register, self._oracle.outcome_register)
        qc.u3(pi, 0, pi, self._oracle.outcome_register)  # x
        qc.u2(0, pi, self._oracle.outcome_register)  # h
        qc += self._init_state_circuit
        qc += self._qc_amplitude_amplification
        self._ret['circuit'] = qc
        return qc

    def _run(self):
        if self._incremental:
            current_max_num_iterations, lam = 1, 6 / 5

            def _try_current_max_num_iterations():
                target_num_iterations = np.random.randint(current_max_num_iterations) + 1
                self._qc_amplitude_amplification = QuantumCircuit()
                for _ in range(target_num_iterations):
                    self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
                return self._run_with_existing_iterations()

            while current_max_num_iterations < self._max_num_iterations:
                assignment, oracle_evaluation = _try_current_max_num_iterations()
                if oracle_evaluation:
                    break
                current_max_num_iterations = min(lam * current_max_num_iterations, self._max_num_iterations)
        else:
            self._qc_amplitude_amplification = QuantumCircuit()
            for i in range(self._num_iterations):
                self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
            assignment, oracle_evaluation = self._run_with_existing_iterations()

        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation
        return self._ret
