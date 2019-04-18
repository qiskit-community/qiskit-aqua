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

import math
import numpy as np
from sklearn.utils import shuffle

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.aqua.algorithms.adaptive.vqalgorithm import VQAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.adaptive.qsvm.qsvm_variational import QSVMVariational
from qiskit.aqua.algorithms.adaptive.qsvm import (cost_estimate, return_probabilities)
from qiskit.aqua.utils import (get_feature_dimension, map_label_to_class_name,
                               split_dataset_to_data_and_labels, find_regs_by_name)
from qiskit.aqua.components.initial_states import Custom



logger = logging.getLogger(__name__)


class QNN(QSVMVariational):

    CONFIGURATION = {
        'name': 'QNN',
        'description': 'QNN Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QNN_schema',
            'type': 'object',
            'properties': {
                'override_SPSA_params': {
                    'type': 'boolean',
                    'default': True
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                },
                'minibatch_size': {
                    'type': 'integer',
                    'default': -1
                }
            },
            'additionalProperties': False
        },
        'problems': ['svm_classification'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'SPSA'
                },
            },
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'RYRZ',
                    'depth': 3
                },
            },
        ],
    }

    def __init__(self, optimizer, var_form, training_dataset,
                 test_dataset=None, datapoints=None, max_evals_grouped=1,
                 minibatch_size=-1, callback=None):
        """Initialize the object
        Args:
            training_dataset (dict): {'A': numpy.ndarray, 'B': numpy.ndarray, ...}
            test_dataset (dict): the same format as `training_dataset`
            datapoints (numpy.ndarray): NxD array, N is the number of data and D is data dimension
            optimizer (Optimizer): Optimizer instance
            var_form (VariationalForm): VariationalForm instance
            max_evals_grouped (int): max number of evaluations performed simultaneously.
            callback (Callable): a callback that can access the intermediate data during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of data batch, the index of evaluation,
                                 parameters of variational form, evaluated value.
        Notes:
            We used `label` denotes numeric results and `class` means the name of that class (str).
        """
        self.validate(locals())
        VQAlgorithm.__init__(self, var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._cost_function_wrapper)
        self._optimizer.set_max_evals_grouped(max_evals_grouped)

        self._callback = callback
        if training_dataset is None:
            raise AquaError('Training dataset must be provided')

        self._training_dataset, self._class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        self._label_to_class = {label: class_name for class_name, label
                                in self._class_to_label.items()}
        self._num_classes = len(list(self._class_to_label.keys()))

        if test_dataset is not None:
            self._test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                  self._class_to_label)
        else:
            self._test_dataset = test_dataset

        if datapoints is not None and not isinstance(datapoints, np.ndarray):
            datapoints = np.asarray(datapoints)
        self._datapoints = datapoints
        self._num_qubits = self._var_form._num_qubits
        self._minibatch_size = minibatch_size
        self._eval_count = 0
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        algo_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        override_spsa_params = algo_params.get('override_SPSA_params')
        max_evals_grouped = algo_params.get('max_evals_grouped')
        minibatch_size = algo_params.get('minibatch_size')

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        # If SPSA then override SPSA params as reqd to our predetermined values
        if opt_params['name'] == 'SPSA' and override_spsa_params:
            opt_params['c0'] = 4.0
            opt_params['c1'] = 0.1
            opt_params['c2'] = 0.602
            opt_params['c3'] = 0.101
            opt_params['c4'] = 0.0
            opt_params['skip_calibration'] = True
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        # we do not need the feature map, which needs n qubits to encode the n-dim data
        # instead, we use custom initialization, which uses log(n) qubits to encode the n-dim data.
        # This shows the advantage of qnn over qsvm.
        data_dim = get_feature_dimension(algo_input.training_dataset)
        num_qubits = int(math.log2(data_dim)) # for n-dim, we need log(n) qubits, rather than n qubits.


        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        return cls(optimizer, var_form, algo_input.training_dataset,
                   algo_input.test_dataset, algo_input.datapoints, max_evals_grouped,
                   minibatch_size)

    def normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def construct_circuit(self, x, theta, measurement=False):
        """
        Construct circuit based on data and parameters in variational form.

        Args:
            x (numpy.ndarray): 1-D array with D dimension
            theta ([numpy.ndarray]): list of 1-D array, parameters sets for variational form
            measurement (bool): flag to add measurement
        Returns:
            QuantumCircuit: the circuit
        """
        qr = QuantumRegister(self._num_qubits, name='q')
        cr = ClassicalRegister(self._num_qubits, name='c')
        qc = QuantumCircuit(qr, cr)
        # encode the data with custom initialization (instead of feature map)
        expected_amplitude_vector = self.normalize(x)
        custominput = Custom(self._num_qubits, state_vector=expected_amplitude_vector)
        qc += custominput.construct_circuit('circuit', qr)
        qc += self._var_form.construct_circuit(theta, qr)

        if measurement:
            qc.barrier(qr)
            qc.measure(qr, cr)
        return qc



    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']

    @property
    def ret(self):
        return self._ret

    @ret.setter
    def ret(self, new_value):
        self._ret = new_value

    @property
    def label_to_class(self):
        return self._label_to_class

    @property
    def class_to_label(self):
        return self._class_to_label

    def load_model(self, file_path):
        model_npz = np.load(file_path)
        self._ret['opt_params'] = model_npz['opt_params']

    def save_model(self, file_path):
        model = {'opt_params': self._ret['opt_params']}
        np.savez(file_path, **model)

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def datapoints(self):
        return self._datapoints
