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

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.utils import get_feature_dimension
from ..vqclassification import VQClassification

logger = logging.getLogger(__name__)


class QSVMVariational(VQClassification):

    CONFIGURATION = {
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
                'pluggable_type': 'feature_map',
                'default': {
                    'name': 'SecondOrderExpansion',
                    'depth': 2
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

    def __init__(
            self,
            optimizer=None,
            feature_map=None,
            var_form=None,
            training_dataset=None,
            test_dataset=None,
            datapoints=None,
            max_evals_grouped=1,
            minibatch_size=-1,
            callback=None
    ):
        """Initialize the object
        Args:
            optimizer (Optimizer): The classical optimizer to use.
            feature_map (FeatureMap): The FeatureMap instance to use.
            var_form (VariationalForm): The variational form instance.
            training_dataset (dict): The training dataset, in the format: {'A': np.ndarray, 'B': np.ndarray, ...}.
            test_dataset (dict): The test dataset, in same format as `training_dataset`.
            datapoints (np.ndarray): NxD array, N is the number of data and D is data dimension.
            max_evals_grouped (int): The maximum number of evaluations to perform simultaneously.
            minibatch_size (int): The size of a mini-batch.
            callback (Callable): a callback that can access the intermediate data during the optimization.
                Internally, four arguments are provided as follows the index of data batch, the index of evaluation,
                parameters of variational form, evaluated value.
        Notes:
            We use `label` to denotes numeric results and `class` the class names (str).
        """

        self.validate(locals())
        super().__init__(
            optimizer=optimizer,
            var_form=var_form,
            num_qubits=feature_map.num_qubits,
            training_dataset=training_dataset,
            test_dataset=test_dataset,
            datapoints=datapoints,
            max_evals_grouped=max_evals_grouped,
            minibatch_size=minibatch_size,
            callback=callback
        )

        self._feature_map = feature_map

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

        # Set up feature map
        fea_map_params = params.get(Pluggable.SECTION_KEY_FEATURE_MAP)
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(params)

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        return cls(optimizer, feature_map, var_form, algo_input.training_dataset,
                   algo_input.test_dataset, algo_input.datapoints, max_evals_grouped,
                   minibatch_size)

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
        qc += self._feature_map.construct_circuit(x, qr)
        qc += self._var_form.construct_circuit(theta, qr)

        if measurement:
            qc.barrier(qr)
            qc.measure(qr, cr)
        return qc
