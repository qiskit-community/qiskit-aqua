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
Providing the A factory to compute the expected value
    E_X( (x - shift)^2 )
for a given parameter `shift`.
"""
from qiskit.aqua.components.uncertainty_problems import UncertaintyProblem
from qiskit.aqua.circuits.gates import cry


class IntShiftAFactory(UncertaintyProblem):
    """
    Providing the A factory to compute the expected value
        E_X( (x - y)^2 )
    for a given integer y < 2^k, represented with qubits r_j:
        y = \sum_{j=0}^{k-1} 2^j r_j
    provided as quantum circuit with k registers.
    """

    CONFIGURATION = {
        'name': 'IntShiftAFactory',
        'description': 'Integer-Shift A factory',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'ECEV_schema',
            'type': 'object',
            'properties': {
                'c_approx': {
                    'type': 'number',
                    'default': 0.1
                },
                'i_param': {
                    'type': 'array',
                    'items': {
                        'type': 'integer'
                    },
                    'default': None
                },
                'i_state': {
                    'type': 'array',
                    'items': {
                        'type': 'integer'
                    },
                    'default': None
                },
                'i_objective': {
                    'type': 'integer',
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, uncertainty_model, c_approx, r, prefactor_x=1,
                 prefactor_y=1, i_state=None, i_objective=None):
        """
        @param uncertainty_model CircuitFactory
        @param c_approx Approx factor for sin(cx) = cx
        @param r QuantumCircuit with state representing y
        """
        self._n = uncertainty_model.num_target_qubits
        self._k = len(r.qubits())
        num_qubits = self._k + self._n
        super().__init__(num_qubits + 1)

        self._uncertainty_model = uncertainty_model
        self._c_approx = c_approx
        self._r = r

        i_param = list(range(self._k))
        if i_state is None:
            i_state = list(range(self._k, num_qubits))
        if i_objective is None:
            i_objective = num_qubits

        self._params = {
            'i_param': i_param,
            'i_state': i_state,
            'i_objective': i_objective
        }

        super().validate(locals())

        self.slope_angle_x = self._c_approx * prefactor_x
        self.slope_angle_y = self._c_approx * prefactor_y

    def value_to_estimation(self, value):
        estimator = value / self._c_approx**2
        return estimator

    def estimation_to_value(self, estimator):
        value = estimator * self._c_approx**2
        return value

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None:
            params = self._params

        # get qubits
        q_objective = q[params['i_objective']]

        print("Params:")
        for key, value in self._params.items():
            print(key, value)

        # apply uncertainty model
        # q_uncertainty = q[self._params['i_state'][0]:(self._params['i_state'][-1] + 1)]
        q_uncertainty = q[0:3]
        print(type(q))
        print("type(q_uncertainty):", type(q_uncertainty))
        print("q_uncertainty:", q_uncertainty)
        self._uncertainty_model.build(qc, q_uncertainty, q_ancillas)

        for pow, i in enumerate(self._params['i_state']):
            qc.cry(2 * self.slope_angle_x * 2 ** pow, q[i], q_objective)

        for pow, j in enumerate(self._params['i_param']):
            qc.cry(-2 * self.slope_angle_y * 2 ** pow, q[j], q_objective)
