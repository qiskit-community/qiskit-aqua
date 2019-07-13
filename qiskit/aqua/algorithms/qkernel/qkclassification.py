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
Quantum Kernel Classification algorithm.

Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow & Jay M. Gambetta
See: https://www.nature.com/articles/s41586-019-0980-2

"""

import logging
import numpy as np

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, QuantumAlgorithm, register_pluggable, AquaError

from qkernels.code.qkernel import QKernel

logger = logging.getLogger(__name__)


class QKernelClassification(QuantumAlgorithm):
    CONFIGURATION = {
        'name': 'QKernel.Classification',
        'description': 'QKernel Classification',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKernel_classification_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'multiclass_mode': {
                    'type': 'string',
                    'default': 'one_vs_one',
                },
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
            {'pluggable_type': 'variational_form',
             'default': {
                 'name': 'RYRZ',
                 'depth': 4
             }
             },
            {'pluggable_type': 'feature_map',
             'default': {
                 'name': 'SecondOrderExpansion',
                 'depth': 2
             }
             },
        ],
    }

    def __init__(self, circuit_maker, X, y, qkernel=None, multiclass_mode='one_vs_one', model_kwargs=None):
        """Constructor.

        Args:
            circuit_maker (VariationalForm or FeatureMap): variational form or feature map, used to transform data

        Raises:
            ValueError: if training_dataset is None
        """
        super().__init__()

        if not qkernel:
            self.qkernel = QKernel(construct_circuit_fn=circuit_maker.construct_circuit,
                                   num_qubits=circuit_maker.num_qubits)
        else:
            self.qkernel = qkernel

        if X is None:
            raise ValueError('Training samples must be provided')
        self._independent = X

        if y is None:
            raise ValueError('Training labels must be provided')
        self._dependent = y
        self._num_classes = len(set(y))

        self._multiclass_mode = multiclass_mode
        self._model_kwargs = model_kwargs or {}

        self._model = None
        self._return = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        (X, y) = algo_input

        if X is None:
            raise ValueError('Training samples must be provided')
        if y is None:
            raise ValueError('Training labels must be provided')

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        model_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_qubits = model_params.get('num_qubits')
        multiclass_mode = model_params.get('multiclass_mode')

        # Default circuit maker is feature map. If none present, check for variational form.
        if X.shape[1] == num_qubits and 'feature_map' in params:
            circuit_maker_params = params.get(Pluggable.SECTION_KEY_FEATURE_MAP)
            circuit_maker_params['feature_dimension'] = num_qubits
            circuit_maker = get_pluggable_class(PluggableType.FEATURE_MAP,
                                              circuit_maker_params['name']).init_params(params)
        elif 'variational_form' in params:
            circuit_maker_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
            circuit_maker_params['num_qubits'] = num_qubits
            circuit_maker = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                           circuit_maker_params['name']).init_params(params)
        else:
            raise AquaError('No var_form or feature_map specified in dictionary')

        return cls(circuit_maker, X, y, multiclass_mode=multiclass_mode)

    def _run(self):
        if not self.qkernel.kernel_matrix:
            self.qkernel.construct_kernel_matrix(self._independent, quantum_instance=self.quantum_instance)
        kernel = self.qkernel.kernel_matrix
        self._return['overlap_matrix'] = kernel

        if self._num_classes > 2:
            if self._multiclass_mode == 'one_vs_one':
                self._model = OneVsOneClassifier(SVC(kernel='precomputed', **self._model_kwargs))
            elif self._multiclass_mode == 'one_vs_rest':
                self._model = OneVsRestClassifier(SVC(kernel='precomputed', **self._model_kwargs))
            elif self._multiclass_mode == 'output_code':
                self._model = OutputCodeClassifier(SVC(kernel='precomputed', **self._model_kwargs))
            else:
                raise AquaError('Multiclass extension {} not supported'.format(self._multiclass_mode))
        else:
            self._model = SVC(kernel='precomputed', **self._model_kwargs)
        labels = self._model.fit(X=kernel, y=self._dependent).predict(X=kernel)
        score = self._model.score(X=kernel, y=self._dependent)

        self._return['labels'] = labels
        self._return['score'] = score
        if self._num_classes > 2:
            self._return['support_indices'] = {i:est.support_
                                               for (i, est) in enumerate(self._model.estimators_)}
            self._return['support_vectors'] = {i:est.support_vectors_
                                               for (i, est) in enumerate(self._model.estimators_)}
        else:
            self._return['support_indices'] = self._model.support_
            self._return['support_vectors'] = self._model.support_vectors_
        return self._return

    def predict(self, new_x):
        # Must compare to full training set for multiclass (for now)
        if self._num_classes > 2:
            comp_vectors = self._independent
        else:
            comp_vectors = self._model.support_vectors_
        new_samples_kernel_entries = self.qkernel.construct_kernel_matrix(new_x, comp_vectors,
                                                 quantum_instance=self.quantum_instance,
                                                 calculate_diags=True,
                                                 save_as_kernel=False)
        return self._model.predict(new_samples_kernel_entries)

    def score(self, new_x, new_y=None):
        if self._num_classes > 2:
            comp_vectors = self._independent
        else:
            comp_vectors = self._model.support_vectors_
        new_samples_kernel_entries = self.qkernel.construct_kernel_matrix(new_x, comp_vectors,
                                                                          quantum_instance=self.quantum_instance,
                                                                          calculate_diags=True,
                                                                          save_as_kernel=False)
        return self._model.score(X=new_samples_kernel_entries, y=new_y)