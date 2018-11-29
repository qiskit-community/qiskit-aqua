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
Amplitude Estimation Algorithm.
"""

import logging
from collections import OrderedDict
import numpy as np

from qiskit_aqua import QuantumAlgorithm, AquaError
from qiskit_aqua import PluggableType, get_pluggable_class
from qiskit_aqua.algorithms.single_sample import PhaseEstimation
from qiskit_aqua.algorithms.components.iqfts import Standard
from .q_factory import QFactory

logger = logging.getLogger(__name__)


class AmplitudeEstimation(QuantumAlgorithm):

    CONFIGURATION = {
        'name': 'AE',
        'description': 'Amplitude Estimation Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'AE_schema',
            'type': 'object',
            'properties': {
                'num_eval_qubits': {
                    'type': 'integer',
                    'default': 5,
                    'minimum': 1
                },
                'additional_params': {
                    'type': ['object', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['amplitude_estimation'],
        'depends': ['problem', 'distribution', 'iqft'],
        'defaults': {
            'distribution': {
                'name': 'NORMAL'
            },
            'iqft': {
                'name': 'STANDARD'
            }
        }
    }

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: Input instance
        """
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        ae_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_eval_qubits = ae_params.get('num_eval_qubits')
        additional_params = ae_params.get('additional_params')

        # Set up distribution
        distribution = None
        """ TODO
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)
        """

        # Set up iqft, we need to add num qubits to params which is our num_ancillae bits here
        iqft_params = params.get(QuantumAlgorithm.SECTION_KEY_IQFT)
        iqft_params['num_qubits'] = num_eval_qubits
        iqft = get_pluggable_class(PluggableType.IQFT, iqft_params['name']).init_params(iqft_params)

        return cls(num_eval_qubits, distribution, q_factory=None, additional_params=additional_params, iqft=iqft)

    def __init__(self, num_eval_qubits, a_factory, q_factory=None, additional_params=None, iqft=None):
        # self.validate(locals())
        super().__init__()

        # get/construct A/Q operator
        self.a_factory = a_factory
        if q_factory is None:
            self.q_factory = QFactory(a_factory)
        else:
            self.q_factory = q_factory

        # get parameters
        self._a_params = additional_params
        self._m = num_eval_qubits
        self._M = 2 ** num_eval_qubits

        # determine number of ancillas
        self._num_ancillas = self.q_factory.required_ancillas_controlled()
        self._num_qubits = self.a_factory.num_target_qubits + self._m + self._num_ancillas

        if iqft is None:
            iqft = Standard(self._m)

        self._iqft = iqft

    def get_circuit(self):

        pe = PhaseEstimation(None, None, self._iqft, num_ancillae=self._m,
                             state_in_circuit_factory=self.a_factory,
                             operator_circuit_factory=self.q_factory,
                             additional_params=self._a_params)

        qc = pe.construct_circuit()

        return qc

        # run circuit
        # qp = QuantumProgram()
        # qp.add_circuit('ae', qc)
        # results = qp.execute('ae', shots=1, timeout=10000, backend='local_statevector_simulator_cpp')
        # state_vector = results.get_statevector()

    def evaluate_results(self, probabilities):

        # map measured results to estimates
        y_probabilities = OrderedDict()
        for i, probability in enumerate(probabilities):
            b = "{0:b}".format(i).rjust(self._num_qubits, '0')[::-1]
            y = int(b[:self._m], 2)
            y_probabilities[y] = y_probabilities.get(y, 0) + probability

        a_probabilities = OrderedDict()
        for y, probability in y_probabilities.items():
            if y >= int(self._M / 2):
                y = self._M - y
            a = np.power(np.sin(y * np.pi / 2 ** self._m), 2)
            a_probabilities[a] = a_probabilities.get(a, 0) + probability

        return a_probabilities, y_probabilities

    # TODO: implement the run method!
    def run(self):
        pass
