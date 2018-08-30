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

from qiskit_aqua.algorithms.many_sample.qsvm import SVM_QKernel_Binary, SVM_QKernel_Multiclass
from qiskit_aqua import (QuantumAlgorithm, get_multiclass_extension_instance,
                         get_feature_map_instance)

logger = logging.getLogger(__name__)


class SVM_QKernel(QuantumAlgorithm):
    """
    The qkernel interface.
    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """
    SVM_QKERNEL_CONFIGURATION = {
        'name': 'QSVM.Kernel',
        'description': 'SVM_QKernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'SVM_QKernel_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'depends': ['multiclass_extension', 'feature_map'],
        'problems': ['svm_classification'],
        'defaults': {
            'multiclass_extension': {
                'name': 'AllPairs',
                'estimator': 'QKernalSVM_Estimator'
            },
            'feature_map': {
                'name': 'SecondOrderExpansion',
                'depth': 2
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SVM_QKERNEL_CONFIGURATION.copy())
        self._ret = {}

    @staticmethod
    def _check_num_classes(dataset):
        """Check number of classes in a given dataset

        Args:
            dataset(dict): key is the class name and value is the data.

        Returns:
            int: number of classes
        """
        return len(list(dataset.keys()))

    @staticmethod
    def _check_feature_dim(dataset):
        """Check number of classes in a given dataset

        Args:
            dataset(dict): key is the class name and value is the data.

        Returns:
            int: number of classes
        """
        for v in dataset.values():
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            return v.shape[1]

    def init_params(self, params, algo_input):
        svmqk_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)

        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        feature_map = get_feature_map_instance(fea_map_params['name'])
        num_qubits = self._check_feature_dim(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map.init_params(fea_map_params)

        is_multiclass = self._check_num_classes(algo_input.training_dataset) > 2

        if is_multiclass:
            multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION)
            multiclass_extension = get_multiclass_extension_instance(multiclass_extension_params['name'])
            # we need to set this explicitly for quantum version
            multiclass_extension_params['params'] = [self._backend, self._execute_config['shots'], self._random_seed]
            multiclass_extension.init_params(multiclass_extension_params)
            # checking the options:
            estimator = multiclass_extension_params.get('estimator', None)
            if estimator is None:
                logger.warning("You did not provide the estimator, which is however required!")
            if estimator not in ["QKernalSVM_Estimator"]:
                logger.warning("You should use one of the qkernel estimators")
            logger.info("We will apply the multiclass classifcation:" + multiclass_extension_params['name'])

            # self.instance = SVM_QKernel_Multiclass(multiclass_extension)
        else:
            logger.warning("Only two classes in the dataset, we will apply the binary classifcation"
                           " and ignore all options related to the multiclass")
            multiclass_extension = None
            # self.instance = SVM_QKernel_Binary()

        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, feature_map, multiclass_extension)

    def init_args(self, training_dataset, test_dataset, datapoints,
                  feature_map, multiclass_extension=None):

        if multiclass_extension is None:
            qsvm_instance = SVM_QKernel_Binary()
        else:
            qsvm_instance = SVM_QKernel_Multiclass(multiclass_extension)
        qsvm_instance.init_args(training_dataset, test_dataset, datapoints, feature_map, self)
        self.instance = qsvm_instance

    def run(self):
        return self.instance.run()
