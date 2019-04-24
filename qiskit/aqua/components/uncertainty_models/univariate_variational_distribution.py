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


import numpy as np

from qiskit.aqua.components.uncertainty_models.univariate_distribution import UnivariateDistribution
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance

CONFIGURATION = {
    'name': 'UnivariateVariationalDistribution',
    'description': 'Uniivariate Variational Distribution',
    'input_schema': {
        '$schema': 'http://json-schema.org/schema#',
        'id': 'UniivariateVariationalDistribution_schema',
        'type': 'object',
        'properties': {
            'num_qubits': {
                'type': 'number',
            },

            'params': {
                'type': 'array',
                "items": {
                    "type": "number"
                }
            },
            'low': {
                'type': 'number',
                'default': 0
            },
            'high': {
                'type': 'number',
                'default': 1
            },
        },
        'additionalProperties': False
    ,
    'depends': [
        {
        'pluggable_type': 'variational_form',
        'default': {'name': 'RY'}
        },
        {'pluggable_type': 'initial_distribution',
                'default': {None}
            },
    ],
}
}

class UnivariateVariationalDistribution(UnivariateDistribution):
    """
    The Univariate Variational Distribution.
    """

    def __init__(self, num_qubits, var_form, params, initial_distribution=None, low=0, high=1):
        self._var_form = var_form
        self.params = params
        self._initial_distribution = initial_distribution
        q_ = QuantumRegister(num_qubits)
        c_ = ClassicalRegister(num_qubits)
        qc_ = QuantumCircuit(q_, c_)
        if not self._initial_distribution is None:
            self._initial_distribution.build(qc_, q_)
        qc_.extend(self._var_form.construct_circuit(self.params, q_))
        qc_.measure(q_,c_)
        quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'))
        result = quantum_instance.execute(qc_)
        result = result.get_statevector(qc_)
        values = np.multiply(result, np.conj(result))
        values = list(values.real)
        probabilities = values
        super().__init__(num_qubits, probabilities, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        if not self._initial_distribution is None:
            self._initial_distribution.build(qc, q, q_ancillas, params)
        qc.extend(self._var_form.construct_circuit(self.params, q))

