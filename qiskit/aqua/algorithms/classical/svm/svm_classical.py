# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The classical svm interface.
"""

import logging

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.classical.svm import (_SVM_Classical_Binary,
                                                  _SVM_Classical_Multiclass,
                                                  _RBF_SVC_Estimator)
from qiskit.aqua.utils import get_num_classes

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class SVM_Classical(QuantumAlgorithm):
    """
    The classical svm interface.
    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """

    CONFIGURATION = {
        'name': 'SVM',
        'description': 'SVM_Classical Algorithm',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'SVM_Classical_schema',
            'type': 'object',
            'properties': {
                'gamma': {
                    'type': ['number', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
            {'pluggable_type': 'multiclass_extension'},
        ],
    }

    def __init__(self, training_dataset, test_dataset=None, datapoints=None,
                 gamma=None, multiclass_extension=None):
        self.validate(locals())
        super().__init__()
        if training_dataset is None:
            raise AquaError('Training dataset must be provided.')

        is_multiclass = get_num_classes(training_dataset) > 2
        if is_multiclass:
            if multiclass_extension is None:
                raise AquaError('Dataset has more than two classes. '
                                'A multiclass extension must be provided.')
        else:
            if multiclass_extension is not None:
                logger.warning("Dataset has just two classes. Supplied multiclass "
                               "extension will be ignored")

        if multiclass_extension is None:
            svm_instance = _SVM_Classical_Binary(training_dataset, test_dataset, datapoints, gamma)
        else:
            svm_instance = _SVM_Classical_Multiclass(
                training_dataset, test_dataset, datapoints, gamma, multiclass_extension)

        self.instance = svm_instance

    @classmethod
    def init_params(cls, params, algo_input):
        """ init params """
        svm_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        gamma = svm_params.get('gamma', None)

        multiclass_extension = None
        multiclass_extension_params = params.get(Pluggable.SECTION_KEY_MULTICLASS_EXT)
        if multiclass_extension_params is not None:
            multiclass_extension_params['estimator_cls'] = _RBF_SVC_Estimator

            multiclass_extension = get_pluggable_class(
                PluggableType.MULTICLASS_EXTENSION,
                multiclass_extension_params['name']).init_params(params)
            logger.info("Multiclass dataset with extension: %s",
                        multiclass_extension_params['name'])

        return cls(algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints, gamma, multiclass_extension)

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

    def _run(self):
        return self.instance.run()

    @property
    def label_to_class(self):
        """ returns label to class """
        return self.instance.label_to_class

    @property
    def class_to_label(self):
        """ returns class to label """
        return self.instance.class_to_label

    @property
    def ret(self):
        """ returns result """
        return self.instance.ret

    @ret.setter
    def ret(self, new_ret):
        """ sets result """
        self.instance.ret = new_ret

    def load_model(self, file_path):
        """Load a model from a file path.

        Args:
            file_path (str): the path of the saved model.
        """
        self.instance.load_model(file_path)

    def save_model(self, file_path):
        """Save the model to a file path.

        Args:
            file_path (str): a path to save the model.
        """
        self.instance.save_model(file_path)
