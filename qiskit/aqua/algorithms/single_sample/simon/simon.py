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
Simon's algorithm.
"""

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class


class Simon(QuantumAlgorithm):
    """The Simon algorithm."""

    CONFIGURATION = {
        'name': 'Simon',
        'description': 'Simon',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'simon_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['periodfinding'],
        'depends': [
            {'pluggable_type': 'oracle',
             'default': {
                     'name': 'simon',
                },
             },
        ],
    }

    def __init__(self, oracle):
        self.validate(locals())
        super().__init__()

        self._oracle = oracle
        self._circuit = None
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        if algo_input is not None:
            raise AquaError("Unexpected Input instance.")

        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        oracle = get_pluggable_class(
            PluggableType.ORACLE,
            oracle_params['name']).init_params(params)
        return cls(oracle)

    def construct_circuit(self):
        if self._circuit is not None:
            return self._circuit

        qc_preoracle = QuantumCircuit(
            self._oracle.variable_register(),
            self._oracle.ancillary_register(),
        )
        qc_preoracle.h(self._oracle.variable_register())
        qc_preoracle.barrier()

        # oracle circuit
        qc_oracle = self._oracle.construct_circuit()
        qc_oracle.barrier()

        # postoracle circuit
        qc_postoracle = QuantumCircuit(
            self._oracle.variable_register(),
            self._oracle.ancillary_register(),
        )
        qc_postoracle.h(self._oracle.variable_register())
        qc_postoracle.barrier()

        # measurement circuit
        measurement_cr = ClassicalRegister(len(
            self._oracle.variable_register()), name='m')

        qc_measurement = QuantumCircuit(
            self._oracle.variable_register(),
            measurement_cr
        )
        qc_measurement.barrier(self._oracle.variable_register())
        qc_measurement.measure(
            self._oracle.variable_register(), measurement_cr)

        self._circuit = qc_preoracle+qc_oracle+qc_postoracle+qc_measurement
        return self._circuit

    def _run(self):
        qc = self.construct_circuit()

        self._ret['circuit'] = qc
        self._ret['measurements'] = self._quantum_instance.execute(
            qc).get_counts(qc)
        self._ret['result'] = self._oracle.interpret_measurement(
            self._ret['measurements'])
        self._ret['oracle_evaluation'] = self._oracle.evaluate_classically(
            self._ret['result'])

        return self._ret
