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

from functools import partial

import numpy as np

from qiskit_acqua import QuantumAlgorithm, get_optimizer_instance
from .cost_helpers import assign_label, cost_estimate_sigmoid, return_probabilities
from .quantum_circuit_variational import eval_cost_function, eval_cost_function_with_unlabeled_data
from .quantum_circuit_variational import set_print_info


def entangler_map_creator(n):
    # helper
    if n == 2:
        entangler_map = {0: [1]}
    elif n == 3:
        entangler_map = {0: [2, 1],
                         1: [2]}
    elif n == 4:
        entangler_map = {0: [2, 1],
                         1: [2],
                         3: [2]}
    elif n == 5:
        entangler_map = {0: [2, 1],
                         1: [2],
                         3: [2, 4],
                         4: [2]}
    return entangler_map


class SVM_Variational(QuantumAlgorithm):

    SVM_VARIATIONAL_CONFIGURATION = {
        'name': 'SVM_Variational',
        'description': 'SVM_Variational Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_Variational_schema',
            'type': 'object',
            'properties': {
                'num_of_qubits': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 2
                },
                'circuit_depth': {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 3
                },
                'print_info': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'depends': ['optimizer'],
        'problems': ['svm_classification'],
        'defaults': {
            'optimizer': {
                'name': 'SPSA'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_VARIATIONAL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        SVMQK_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        self.training_dataset = algo_input.training_dataset
        self.test_dataset = algo_input.test_dataset
        self.datapoints = algo_input.datapoints
        self.class_labels = list(self.training_dataset.keys())

        num_of_qubits = SVMQK_params.get('num_of_qubits')
        circuit_depth = SVMQK_params.get('circuit_depth')
        print_info = SVMQK_params.get('print_info')

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_optimizer_instance(opt_params['name'])
        # hard-coded params if SPSA is used.
        if opt_params['name'] == 'SPSA' and opt_params['parameters'] is None:
            opt_params['parameters'] = np.asarray([4.0, 0.1, 0.602, 0.101, 0.0])
        optimizer.init_params(opt_params)
        optimizer.set_options(save_steps=10)

        self.init_args(num_of_qubits, circuit_depth, print_info, optimizer)

    def init_args(self, num_of_qubits=2, circuit_depth=3, print_info=False, optimizer=None):
        self.num_of_qubits = num_of_qubits
        self.entangler_map = entangler_map_creator(num_of_qubits)
        self.coupling_map = None  # the coupling_maps gates allowed on the device
        self.initial_layout = None
        self.shots = self._execute_config['shots']
        self.backend = self._backend
        self.circuit_depth = circuit_depth
        self.print_info = print_info
        self.optimizer = optimizer

        set_print_info(print_info)

    def train(self, training_input, class_labels):
        initial_theta = np.random.randn(2 * self.num_of_qubits * (self.circuit_depth + 1))

        eval_cost_function_partial = partial(eval_cost_function, self.entangler_map, self.coupling_map,
                                             self.initial_layout, self.num_of_qubits, self.circuit_depth,
                                             training_input, class_labels, self.backend, self.shots, True)

        def objective_function(theta):
            return eval_cost_function_partial(theta)[0]

        theta_best, cost_final, _ = self.optimizer.optimize(
            initial_theta.shape[0], objective_function, initial_point=initial_theta)
        costs = [cost_final]  # , cost_plus, cost_minus
        return theta_best, costs

    def test(self, theta_best, test_input, class_labels):
        total_cost, std_cost, success_ratio, predicted_labels = \
            eval_cost_function(self.entangler_map, self.coupling_map, self.initial_layout, self.num_of_qubits,
                               self.circuit_depth, test_input, class_labels, self.backend, self.shots,
                               train=False, theta=theta_best)

        if self.print_info:
            print('Classification success for this set is  %s %%  \n' % (100.0 * success_ratio))
        return success_ratio

    def predict(self, theta_best, input_datapoints, class_labels):
        predicted_labels = eval_cost_function_with_unlabeled_data(self.entangler_map, self.coupling_map,
                                                                  self.initial_layout, self.num_of_qubits,
                                                                  self.circuit_depth, input_datapoints, class_labels,
                                                                  self.backend, self.shots,
                                                                  train=False, theta=theta_best)
        return predicted_labels

    def run(self):
        if self.training_dataset is None:
            self._ret['error'] = 'training dataset is missing! please provide it'
            return self._ret

        theta_best, costs = self.train(self.training_dataset, self.class_labels)

        if self.test_dataset is not None:
            success_ratio = self.test(theta_best, self.test_dataset, self.class_labels)
            self._ret['test_success_ratio'] = success_ratio

        if self.datapoints is not None:
            predicted_labels = self.predict(theta_best, self.datapoints, self.class_labels)
            self._ret['predicted_labels'] = predicted_labels

        return self._ret
