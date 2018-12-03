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
The Deutsch-Jozsa algorithm.
"""

import logging
from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_oracle_instance
from qiskit import ClassicalRegister, QuantumCircuit 

logger = logging.getLogger(__name__)

class DeutschJozsa(QuantumAlgorithm):
    """The Deutsch-Jozsa algorithm."""
        
    DJ_CONFIGURATION = {
        'name': 'DeutschJozsa',
        'description': 'DeutschJozsa',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dj_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['deutschjozsa'],
        'depends': ['oracle'],
        'defaults': {
            'oracle': {
                'name': 'deutschjozsa'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DJ_CONFIGURATION.copy())
        self._input = {}
        self._function = None
        self._oracle = None
        self._return = {}
        pass

    def init_params(self, params, algo_input):
        dj_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        self._input = algo_input
 
        oracle_params = params.get(QuantumAlgorithm.SECTION_KEY_ORACLE)
        oracle = get_oracle_instance(oracle_params['name'])
        oracle.init_params(oracle_params)
        self.init_oracle(oracle, algo_input)

    def init_oracle(self, oracle, algo_input):
        oracle.check_input(algo_input)
        oracle.construct_circuit(algo_input)
        self._oracle = oracle

    def _construct_circuit_components(self):        
        # preoracle circuit
        qc_preoracle = QuantumCircuit(
            self._oracle.variable_register(),
            self._oracle.ancillary_register(),
        )
        qc_preoracle.h(self._oracle.variable_register())        
        qc_preoracle.x(self._oracle.ancillary_register())        
        qc_preoracle.h(self._oracle.ancillary_register())                
        qc_preoracle.barrier()
        
        # oracle circuit
        qc_oracle = self._oracle.circuit()
        qc_oracle.barrier()

        # postoracle circuit
        qc_postoracle = QuantumCircuit(
            self._oracle.variable_register(),
            self._oracle.ancillary_register(),
        )
        qc_postoracle.h(self._oracle.variable_register())        
        qc_postoracle.barrier()

        # measurement circuit
        measurement_cr = ClassicalRegister(len(self._oracle.variable_register()), name='m')

        qc_measurement = QuantumCircuit(
            self._oracle.variable_register(),
            measurement_cr
        )
        qc_measurement.barrier(self._oracle.variable_register())
        qc_measurement.measure(self._oracle.variable_register(), measurement_cr)
        
        qc = qc_preoracle + qc_oracle + qc_postoracle + qc_measurement
        return qc
        
    def run(self):
        qc = self._construct_circuit_components()
        
        self._return['circuit'] = qc
        self._return['measurements'] = self.execute(qc).get_counts(qc)
        self._return['result'] = self._oracle.interpret_measurement(self._return['measurements'])
        self._return['oracle_evaluation'] = self._oracle.evaluate_classically(self._return['result'])
        
        return self._return