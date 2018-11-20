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

from qiskit_aqua import (AlgorithmError, QuantumAlgorithm, PluggableType, get_pluggable_class)
from qiskit_aqua.algorithms.many_sample.qsvm import (QSVM_Kernel_Binary, QSVM_Kernel_Multiclass,
                                                     QSVM_Kernel_Estimator)
from qiskit_aqua.utils.dataset_helper import get_feature_dimension, get_num_classes


logger = logging.getLogger(__name__)


class QSVM_Kernel(QuantumAlgorithm):
    """
    Quantum SVM kernel method.

    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """

    CONFIGURATION = {
        'name': 'QSVM.Kernel',
        'description': 'QSVM_Kernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QSVM_Kernel_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'depends': ['multiclass_extension', 'feature_map'],
        'problems': ['svm_classification'],
        'defaults': {
            'feature_map': {
                'name': 'SecondOrderExpansion',
                'depth': 2
            }
        }
    }

    def __init__(self):
        super().__init__(self.CONFIGURATION.copy())
        self._ret = {}
        self.instance = None

    def init_params(self, params, algo_input):

        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,fea_map_params['name'])
        feature_map = feature_map()
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map.init_params(fea_map_params)

        multiclass_extension = None
        multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION)
        if multiclass_extension_params is not None:
            multiclass_extension = get_pluggable_class(PluggableType.MULTICLASS_EXTENSION,multiclass_extension_params['name'])
            multiclass_extension = multiclass_extension()
            multiclass_extension_params['params'] = [feature_map, self]
            multiclass_extension_params['estimator_cls'] = QSVM_Kernel_Estimator
            multiclass_extension.init_params(multiclass_extension_params)
            logger.info("Multiclass dataset with extension: {}".format(multiclass_extension_params['name']))

        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, feature_map, multiclass_extension)

    def init_args(self, training_dataset, test_dataset, datapoints,
                  feature_map, multiclass_extension=None):

    def __init__(self, training_dataset, test_dataset=None, datapoints=None,
                 feature_map=None, multiclass_extension=None):
        """Constructor.

        Args:
            training_dataset (dict)
            test_dataset (dict)
            datapoints (numpy.ndarray)
            feature_map (FeatureMap)
            multiclass_extension (MultiExtension)

        Raises:
            ValueError: if training_dataset is None
            AlgorithmError: use binary classifer for classes > 3
        """
        super().__init__(self.CONFIGURATION.copy())
        if training_dataset is None:
            raise ValueError('Training dataset must be provided')

        is_multiclass = get_num_classes(training_dataset) > 2
        if is_multiclass:
            if multiclass_extension is None:
                raise AlgorithmError('Dataset has more than two classes. A multiclass extension must be provided.')
        else:
            if multiclass_extension is not None:
                logger.warning("Dataset has just two classes. Supplied multiclass extension will be ignored")

        if multiclass_extension is None:
            qsvm_instance = QSVM_Kernel_Binary()
        else:
            qsvm_instance = QSVM_Kernel_Multiclass(multiclass_extension)

        qsvm_instance.init_args(training_dataset, test_dataset, datapoints, feature_map, self)
        self.instance = qsvm_instance
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):

        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP, fea_map_params['name'])
        feature_map = feature_map()
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map.init_params(fea_map_params)

        multiclass_extension = None
        multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION)
        if multiclass_extension_params is not None:
            multiclass_extension = get_pluggable_class(
                PluggableType.MULTICLASS_EXTENSION, multiclass_extension_params['name'])
            multiclass_extension = multiclass_extension()
            multiclass_extension_params['params'] = [feature_map, self]
            multiclass_extension_params['estimator_cls'] = QSVM_Kernel_Estimator
            multiclass_extension.init_params(multiclass_extension_params)
            logger.info("Multiclass dataset with extension: {}".format(multiclass_extension_params['name']))

        return cls(algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints, feature_map, multiclass_extension)

    def train(self, data, labels):
        """
        train the svm
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        """
        self.instance.train(data, labels)

    def test(self, data, labels):
        """
        test the svm
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data

        Returns:
            float: accuracy
        """
        return self.instance.test(data, labels)

    def predict(self, data):
        """
        predict using the svm
        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        return self.instance.predict(data)

    def run(self):
        return self.instance.run()

    @property
    def label_to_class(self):
        return self.instance.label_to_class

    @property
    def class_to_label(self):
        return self.instance.class_to_label

    @property
    def ret(self):
        return self.instance.ret
