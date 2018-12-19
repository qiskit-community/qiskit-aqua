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

from qiskit_aqua.algorithms import QuantumAlgorithm
from qiskit_aqua import AquaError, PluggableType, get_pluggable_class
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_binary import _QSVM_Kernel_Binary
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_multiclass import _QSVM_Kernel_Multiclass
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_estimator import _QSVM_Kernel_Estimator
from qiskit_aqua.utils.dataset_helper import get_feature_dimension, get_num_classes

logger = logging.getLogger(__name__)


class QSVMKernel(QuantumAlgorithm):
    """
    Quantum SVM kernel method.

    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """

    CONFIGURATION = {
        'name': 'QSVM.Kernel',
        'description': 'QSVMKernel Algorithm',
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

    def __init__(self, feature_map, training_dataset, test_dataset=None, datapoints=None,
                 multiclass_extension=None):
        """Constructor.

        Args:
            feature_map (FeatureMap): feature map module, used to transform data
            training_dataset (dict): training dataset.
            test_dataset (dict): testing dataset.
            datapoints (numpy.ndarray): prediction dataset.
            multiclass_extension (MultiExtension): if number of classes > 2, a multiclass scheme is
                                                    is needed.

        Raises:
            ValueError: if training_dataset is None
            AquaError: use binary classifer for classes > 3
        """
        super().__init__()
        if training_dataset is None:
            raise ValueError('Training dataset must be provided')

        is_multiclass = get_num_classes(training_dataset) > 2
        if is_multiclass:
            if multiclass_extension is None:
                raise AquaError('Dataset has more than two classes. A multiclass extension must be provided.')
        else:
            if multiclass_extension is not None:
                logger.warning("Dataset has just two classes. Supplied multiclass extension will be ignored")

        if multiclass_extension is None:
            qsvm_instance = _QSVM_Kernel_Binary(feature_map, self, training_dataset, test_dataset, datapoints)
        else:
            qsvm_instance = _QSVM_Kernel_Multiclass(
                feature_map, self, training_dataset, test_dataset, datapoints, multiclass_extension)

        self.instance = qsvm_instance

    @classmethod
    def init_params(cls, params, algo_input):
        """
        """
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        fea_map_params['num_qubits'] = num_qubits

        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(fea_map_params)

        multiclass_extension = None
        multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION, None)
        if multiclass_extension_params is not None:
            multiclass_extension_params['params'] = [feature_map]
            multiclass_extension_params['estimator_cls'] = _QSVM_Kernel_Estimator

            multiclass_extension = get_pluggable_class(PluggableType.MULTICLASS_EXTENSION,
                                                       multiclass_extension_params['name']).init_params(multiclass_extension_params)
            logger.info("Multiclass classifier based on {}".format(multiclass_extension_params['name']))

        return cls(feature_map, algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints, multiclass_extension)

    def train(self, data, labels, quantum_instance=None):
        """
        Train the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        """
        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        self.instance.train(data, labels)

    def test(self, data, labels, quantum_instance=None):
        """
        Test the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            float: accuracy
        """
        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        return self.instance.test(data, labels)

    def predict(self, data, quantum_instance=None):
        """
        Predict using the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        self._quantum_instance = self._quantum_instance if quantum_instance is None else quantum_instance
        return self.instance.predict(data)

    def _run(self):
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

    @ret.setter
    def ret(self, new_value):
        self.instance.ret = new_value

    def load_model(self, file_path):
        self.instance.load_model(file_path)

    def save_model(self, file_path):
        self.instance.save_model(file_path)

    @property
    def test_dataset(self):
        return self.instance.test_dataset

    @property
    def train_dataset(self):
        return self.instance.train_dataset

    @property
    def datapoints(self):
        return self.instance.datapoints
