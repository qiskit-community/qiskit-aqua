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

import logging

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit_aqua import (AlgorithmError, QuantumAlgorithm,
                         get_feature_map_instance, get_optimizer_instance,
                         get_variational_form_instance)
from qiskit_aqua.algorithms.adaptive.qsvm import (cost_estimate_sigmoid, return_probabilities)
from qiskit_aqua.utils import (get_feature_dimension, map_label_to_class_name,
                               split_dataset_to_data_and_labels)

logger = logging.getLogger(__name__)


class QSVMVariational(QuantumAlgorithm):

    QSVM_VARIATIONAL_CONFIGURATION = {
        'name': 'QSVM.Variational',
        'description': 'QSVM_Variational Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_Variational_schema',
            'type': 'object',
            'properties': {
                'override_SPSA_params': {
                    'type': 'boolean',
                    'default': True
                }
            },
            'additionalProperties': False
        },
        'depends': ['optimizer', 'feature_map', 'variational_form'],
        'problems': ['svm_classification'],
        'defaults': {
            'optimizer': {
                'name': 'SPSA'
            },
            'feature_map': {
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
        super().__init__(configuration or self.QSVM_VARIATIONAL_CONFIGURATION.copy())
        self._ret = {}

    def init_params(self, params, algo_input):
        algo_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        override_spsa_params = algo_params.get('override_SPSA_params')

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_optimizer_instance(opt_params['name'])
        # If SPSA then override SPSA params as reqd to our predetermined values
        if opt_params['name'] == 'SPSA' and override_spsa_params:
            opt_params['c0'] = 4.0
            opt_params['c1'] = 0.1
            opt_params['c2'] = 0.602
            opt_params['c3'] = 0.101
            opt_params['c4'] = 0.0
            opt_params['skip_calibration'] = True
        optimizer.init_params(opt_params)

        # Set up variational form
        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map = get_feature_map_instance(fea_map_params['name'])
        feature_map.init_params(fea_map_params)

        # Set up variational form
        var_form_params = params.get(QuantumAlgorithm.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = num_qubits
        var_form = get_variational_form_instance(var_form_params['name'])
        var_form.init_params(var_form_params)

        self.init_args(algo_input.training_dataset, algo_input.test_dataset, algo_input.datapoints,
                       optimizer, feature_map, var_form)

    def init_args(self, training_dataset, test_dataset, datapoints, optimizer,
                  feature_map, var_form):
        """Initialize the object
        Args:
            training_dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
            test_dataset (dict): the same format as `training_dataset`
            datapoints (numpy.ndarray): NxD array, N is the number of data and D is data dimension
            optimizer (Optimizer): Optimizer instance
            feature_map (FeatureMap): FeatureMap instance
            var_form (VariationalForm): VariationalForm instance
        Notes:
            We used `label` denotes numeric results and `class` means the name of that class (str).
        """

        if 'statevector' in self._backend:
            raise ValueError('Selected backend  "{}" is not supported.'.format(self._backend))

        if training_dataset is None:
            raise AlgorithmError('Training dataset must be provided')

        self._training_dataset, self._class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        if test_dataset is not None:
            self._test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                  self._class_to_label)

        self._label_to_class = {label: class_name for class_name, label
                                in self._class_to_label.items()}
        self._num_classes = len(list(self._class_to_label.keys()))

        self._datapoints = datapoints
        self._optimizer = optimizer
        self._feature_map = feature_map
        self._var_form = var_form
        self._num_qubits = self._feature_map.num_qubits

    def _construct_circuit(self, x, theta):
        qr = QuantumRegister(self._num_qubits, name='q')
        cr = ClassicalRegister(self._num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)
        qc += self._feature_map.construct_circuit(x, qr)
        qc += self._var_form.construct_circuit(theta, qr)
        qc.measure(qr, cr)
        return qc

    def _cost_function(self, predicted_probs, labels):
        """
        Calculate cost of predicted probability of ground truth label based on sigmoid function,
        and the accuracy
        Args:
            predicted_probs (numpy.ndarray): NxK array
            labels (numpy.ndarray): Nx1 array
        Returns:
            float: cost
        """
        total_loss = cost_estimate_sigmoid(self._execute_config['shots'], predicted_probs, labels)
        return total_loss

    def _get_prediction(self, data, theta):
        """
        Make prediction on data based the theta.
        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta (numpy.ndarray): 1-D array, parameters for variational form
        Returns:
            numpy.ndarray: NxK array
            numpy.ndarray: Nx1 array
        """
        predicted_probs = []
        predicted_labels = []
        circuits = {}
        for c_id, datum in enumerate(data):
            circuit = self._construct_circuit(datum, theta)
            circuits[c_id] = circuit

        results = self.execute(list(circuits.values()))

        counts = []
        for c_id, result in enumerate(results):
            counts.append(results.get_counts(circuits[c_id]))
        predicted_probs = return_probabilities(counts, self._num_classes)
        predicted_labels = np.argmax(predicted_probs, axis=1)

        return predicted_probs, predicted_labels

    def train(self, data, labels):
        """Train the models, and save results
        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
        """
        def cost_function_wrapper(theta):
            predicted_probs, predicted_labels = self._get_prediction(data, theta)
            total_cost = self._cost_function(predicted_probs, labels)
            return total_cost

        initial_theta = self.random.randn(self._var_form.num_parameters)

        theta_best, cost_final, _ = self._optimizer.optimize(
            initial_theta.shape[0], cost_function_wrapper, initial_point=initial_theta)

        self._ret['opt_params'] = theta_best
        self._ret['training_loss'] = cost_final

    def test(self, data, labels):
        """Predict the labels for the data, and test against with ground truth labels
        Args:
            data (numpy.ndarray): NxD array, N is number of data and D is data dimension
            labels (numpy.ndarray): Nx1 array, N is number of data
        Returns:
            float: classification accuracy
        """
        predicted_probs, predicted_labels = self._get_prediction(data, self._ret['opt_params'])
        total_cost = self._cost_function(predicted_probs, labels)
        accuracy = np.sum((np.argmax(predicted_probs, axis=1) == labels)) / labels.shape[0]
        logger.debug('Accuracy is {:.2f}%  \n'.format(accuracy * 100.0))
        self._ret['testing_accuracy'] = accuracy
        self._ret['test_success_ratio'] = accuracy
        self._ret['testing_loss'] = total_cost
        return accuracy

    def predict(self, data):
        """Predict the labels for the data
        Args:
            data (numpy.ndarray): NxD array, N is number of data, D is data dimension
        Returns:
            [dict]: for each data point, generates the predicted probability for each class
            list: for each data point, generates the predicted label, which with the highest prob
        """
        predicted_probs, predicted_labels = self._get_prediction(data, self._ret['opt_params'])
        self._ret['predicted_probs'] = predicted_probs
        self._ret['predicted_labels'] = predicted_labels
        return predicted_probs, predicted_labels

    def run(self):
        self.train(self._training_dataset[0], self._training_dataset[1])

        if self._test_dataset is not None:
            self.test(self._test_dataset[0], self._test_dataset[1])

        if self._datapoints is not None:
            predicted_probs, predicted_labels = self.predict(self._datapoints)
            self._ret['predicted_classes'] = map_label_to_class_name(predicted_labels,
                                                                     self._label_to_class)

        return self._ret

    @property
    def ret(self):
        return self._ret

    @property
    def label_to_class(self):
        return self._label_to_class

    @property
    def class_to_label(self):
        return self._class_to_label
