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
import logging
import operator

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from qiskit_aqua import (QuantumAlgorithm, AlgorithmError,
                         get_optimizer_instance, get_feature_extraction_instance,
                         get_variational_form_instance)
from qiskit_aqua.svm import (cost_estimate_sigmoid, return_probabilities)

logger = logging.getLogger(__name__)


class SVM_Variational(QuantumAlgorithm):

    SVM_VARIATIONAL_CONFIGURATION = {
        'name': 'SVM_Variational',
        'description': 'SVM_Variational Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_Variational_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'depends': ['optimizer', 'feature_extraction', 'variational_form'],
        'problems': ['svm_classification'],
        'defaults': {
            'optimizer': {
                'name': 'SPSA'
            },
            'feature_extraction': {
                'name': 'SecondOrderExpansion',
                'depth': 2
            },
            'variational_form': {
                'name': 'RYRZ',
                'depth': 3
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_VARIATIONAL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_optimizer_instance(opt_params['name'])
        # hard-coded params if SPSA is used.
        if opt_params['name'] == 'SPSA' and opt_params['parameters'] is None:
            opt_params['parameters'] = np.asarray([4.0, 0.1, 0.602, 0.101, 0.0])
        optimizer.init_params(opt_params)

        # Set up variational form
        fea_ext_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_EXTRACTION)
        num_qubits = self._auto_detect_qubitnum(algo_input.training_dataset)
        fea_ext_params['num_qubits'] = num_qubits
        feature_extraction = get_feature_extraction_instance(fea_ext_params['name'])
        feature_extraction.init_params(fea_ext_params)

        # Set up variational form
        var_form_params = params.get(QuantumAlgorithm.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = num_qubits
        var_form = get_variational_form_instance(var_form_params['name'])
        var_form.init_params(var_form_params)

        self.init_args(algo_input.training_dataset, algo_input.test_dataset, algo_input.datapoints,
                       optimizer, feature_extraction, var_form)

    def init_args(self, training_dataset, test_dataset, datapoints, optimizer,
                  feature_extraction, var_form):

        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" does not support measurements.'.format(self._backend))

        if training_dataset is None:
            raise ValueError('training dataset is missing! please provide it')

        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.datapoints = datapoints
        self.class_labels = list(self.training_dataset.keys())
        self.class_to_label = {name: idx for idx, name in enumerate(self.class_labels)}

        self.optimizer = optimizer
        self.feature_extraction = feature_extraction
        self.var_form = var_form

        self.num_qubits = self._auto_detect_qubitnum(training_dataset)

    def _auto_detect_qubitnum(self, training_dataset):
        auto_detected_size = -1
        for key, val in training_dataset.items():
            for item in val:
                auto_detected_size = len(item)
                break
        if auto_detected_size == -1:
            raise AlgorithmError('Something wrong in detecting required qubits, please check your data set.')
        if auto_detected_size != 2 and auto_detected_size != 3:
            raise AlgorithmError('You should lower the feature dimension to 2 or 3 first!')
        return auto_detected_size

    def _construct_circuit(self, x, theta):
        qr = QuantumRegister(self.num_qubits, name='q')
        cr = ClassicalRegister(self.num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)
        qc += self.feature_extraction.construct_circuit(x, qr)
        # qc.barrier(qr)
        qc += self.var_form.construct_circuit(theta, qr)
        # qc.barrier(qr)
        qc.measure(qr, cr)

        return qc

    def _cost_function(self, predicted_probs, labels):
        """
        Calculate cost of predicted probability of ground truth label based on sigmoid function,
        and the accuracy

        Returns:
            float: cost
            float: accuracy
        """
        loss = 0.0
        accuracy = 0.0
        for idx, gt_label in enumerate(labels):
            prob = predicted_probs[idx]
            predicted_class = max(prob.items(), key=operator.itemgetter(1))[0]
            expected_class = self.class_labels[gt_label]
            loss += cost_estimate_sigmoid(self._execute_config['shots'], prob, expected_class)

            if predicted_class == expected_class:
                accuracy += 1.0

        accuracy = accuracy / len(labels)
        total_loss = loss / len(labels)

        return total_loss, accuracy

    def _get_prediction(self, data, theta):
        """
        Make prediction on data based the theta.

        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta (numpy.ndarray): 1-D array, parameters for variational form

        Returns:
            [dict]: for each data point, generates the predicted probability for each class
            list: for each data point, generates the predicted label, which with the highest prob
        """
        predicted_probs = []
        predicted_labels = []
        circuits = {}
        for c_id, datum in enumerate(data):
            circuit = self._construct_circuit(datum, theta)
            circuits[c_id] = circuit

        results = self.execute(list(circuits.values()))

        for c_id, result in enumerate(results):
            counts = results.get_counts(circuits[c_id])
            prob = return_probabilities(counts, self.class_labels)
            predicted_probs.append(prob)
            predicted_labels.append(max(prob.items(), key=operator.itemgetter(1))[0])

        return predicted_probs, predicted_labels

    def train(self, data):
        data_samples = []
        labels = []
        for key, values in data.items():
            for value in values:
                data_samples.append(value)
                labels.append(self.class_to_label[key])

        def cost_function_wrapper(theta):
            predicted_probs, predicted_labels = self._get_prediction(data_samples, theta)
            total_cost, accuracy = self._cost_function(predicted_probs, labels)
            return total_cost

        initial_theta = self.random.randn(self.var_form.num_parameters)

        theta_best, cost_final, _ = self.optimizer.optimize(
            initial_theta.shape[0], cost_function_wrapper, initial_point=initial_theta)

        self._ret['opt_params'] = theta_best
        self._ret['training_loss'] = cost_final


    def test(self, data):
        data_samples = []
        labels = []
        for key, values in data.items():
            for value in values:
                data_samples.append(value)
                labels.append(self.class_to_label[key])

        predicted_probs, predicted_labels = self._get_prediction(data_samples, self._ret['opt_params'])
        total_cost, accuracy = self._cost_function(predicted_probs, labels)
        logger.debug('Classification success for this set is  {:.2f}%  \n'.format(accuracy * 100.0))
        self._ret['accuracy'] = accuracy
        return accuracy

    def predict(self, data):
        predicted_probs, predicted_labels = self._get_prediction(data, self._ret['opt_params'])
        return predicted_labels

    def run(self):
        self.train(self.training_dataset)

        if self.test_dataset is not None:
            accuracy = self.test(self.test_dataset)
            self._ret['test_success_ratio'] = accuracy

        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            self._ret['predicted_labels'] = predicted_labels

        return self._ret


if __name__ == '__main__':
    a = SVM_Variational()

