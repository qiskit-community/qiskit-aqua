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

from qiskit_aqua import (AlgorithmError, QuantumAlgorithm, get_feature_map_instance,
                         get_multiclass_extension_instance)
from qiskit_aqua.algorithms.many_sample.qsvm import (SVM_QKernel_Binary, SVM_QKernel_Multiclass,
                                                     QKernalSVM_Estimator)
from qiskit_aqua.utils.dataset_helper import get_feature_dimension, get_num_classes


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
                'name': 'AllPairs'
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

    def init_params(self, params, algo_input):

        if algo_input.training_dataset is None:
            raise AlgorithmError("Training dataset is required.")
        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        feature_map = get_feature_map_instance(fea_map_params['name'])
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params['num_qubits'] = num_qubits
        feature_map.init_params(fea_map_params)

        is_multiclass = get_num_classes(algo_input.training_dataset) > 2

        if is_multiclass:
            multicls_ext_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION)
            multiclass_extension = get_multiclass_extension_instance(multicls_ext_params['name'])
            # we need to set this explicitly for quantum version
            multicls_ext_params['params'] = [feature_map, self]
            multicls_ext_params['estimator_cls'] = QKernalSVM_Estimator
            multiclass_extension.init_params(multicls_ext_params)
            logger.info("Multiclass classifcation algo:" + multicls_ext_params['name'])
        else:
            logger.warning("Only two classes in the dataset, use binary classifer"
                           " and ignore all options of multiclass_extension")
            multiclass_extension = None

        self.init_args(algo_input.training_dataset, algo_input.test_dataset,
                       algo_input.datapoints, feature_map, multiclass_extension)

    def init_args(self, training_dataset, test_dataset, datapoints,
                  feature_map, multiclass_extension=None):

        if multiclass_extension is None:
            qsvm_instance = SVM_QKernel_Binary()
        else:
            qsvm_instance = SVM_QKernel_Multiclass(multiclass_extension)

        if datapoints is not None:
            if not isinstance(datapoints, np.ndarray):
                datapoints = np.asarray(datapoints)

        qsvm_instance.init_args(training_dataset, test_dataset, datapoints, feature_map, self)
        self.instance = qsvm_instance

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
