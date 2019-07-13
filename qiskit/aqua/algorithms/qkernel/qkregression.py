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
Quantum Kernel Regression algorithms.

Donny Greenberg & Anna Phan (contributed equally)
Publication pending.

"""

import logging
import numpy as np
import copy

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, QuantumAlgorithm, register_pluggable
from qiskit.aqua import AquaError

from qiskit.aqua.algorithms.qkernel import QKernel

logger = logging.getLogger(__name__)


class QKernelRegression(QuantumAlgorithm):
    CONFIGURATION = {
        'name': 'QKernel.Regression',
        'description': 'QKernel Regression',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKernel_SVM_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['regression'],
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

    def __init__(self, circuit_maker, X, y, modes='all', mode_kwargs={}, qkernel=None):
        """Constructor.

        Args:
            feature_map (FeatureMap): feature map module, used to transform data
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            y : array-like, shape (n_samples,)

        Raises:
            ValueError: if training_dataset is None
        """
        super().__init__()

        if not qkernel:
            self.qkernel = QKernel(construct_circuit_fn=circuit_maker.construct_circuit,
                                   num_qubits=circuit_maker.num_qubits)
        else:
            self.qkernel = qkernel

        all_modes = ['svr', 'ridge', 'gpr']
        if modes == 'all':
            self.modes = all_modes
            self.mode_kwargs = mode_kwargs
        elif set(modes) <= set(all_modes):
            self.modes = modes
            self.mode_kwargs = mode_kwargs
        else:
            raise ValueError('Regression mode {} not supported'.format(modes))
        self._models = {}

        if X is None:
            raise ValueError('Independent variable dataset must be provided')
        self._independent = X

        if y is None:
            raise ValueError('Dependent variable dataset must be provided')
        self._dependent = y

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

        algo_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_qubits = algo_params.get('num_qubits')

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

        return cls(circuit_maker, X, y)

    def _run(self):

        if self.qkernel.kernel_matrix is None:
            self.qkernel.construct_kernel_matrix(self._independent, quantum_instance=self.quantum_instance)
        kernel = self.qkernel.kernel_matrix

        for mode in self.modes:
            kwargs = self.mode_kwargs.get(mode, {})
            model = None
            if mode == 'svr':
                model = SVR(kernel='precomputed', **kwargs)
            elif mode == 'ridge':
                model = KernelRidge(kernel='precomputed', **kwargs)
            elif mode == 'gpr':
                model = GaussianProcessRegressor(kernel=PairwiseKernel(metric='precomputed'), **kwargs)
            else:
                raise ValueError('Unknown Regression mode {} not supported'.format(mode))
            self._models[mode] = model

            predict = model.fit(kernel, self._dependent).predict(kernel)
            score = model.score(kernel, self._dependent)
            self._return[mode] = {}
            # TODO too heavy to return the whole model?
            # self._return[mode]['model'] = model
            self._return[mode]['predict'] = predict
            self._return[mode]['score'] = score
        self._return['kernel_matrix'] = self.qkernel.kernel_matrix

        return self._return

    def predict(self, new_x, new_y_to_score=None):
        new_samples_kernel_entries = self.qkernel.construct_kernel_matrix(new_x, self._independent,
                                                 quantum_instance=self.quantum_instance,
                                                 calculate_diags=True,
                                                 save_as_kernel=False)
        predictions = {}
        # scores = {}
        errors = {}
        for name, model in self._models.items():
            predictions[name] = model.predict(new_samples_kernel_entries)
            if new_y_to_score is not None:
                # scores[name] = model.score(new_samples_kernel_entries, new_y_to_score)
                errors[name] = np.abs(predictions[name] - new_y_to_score)
        return predictions, errors

    def score_round_robin(self, quantum_instance=None):
        predictions = []
        errors = []
        for i, (x, y) in enumerate(zip(self._independent, self._dependent)):
            new_qk = copy.deepcopy(self.qkernel)
            new_qk.kernel_matrix = np.delete(new_qk.kernel_matrix, i, axis=0)
            new_qk.kernel_matrix = np.delete(new_qk.kernel_matrix, i, axis=1)
            new_ind = np.delete(copy.deepcopy(self._independent), i, axis=0)
            new_dep = np.delete(copy.deepcopy(self._dependent), i, axis=0)
            new_qkr = QKernelRegression(circuit_maker=self.qkernel.construct_circuit_fn,
                                        X=new_ind,
                                        y=new_dep,
                                        qkernel=new_qk,
                                        modes=self.modes,
                                        mode_kwargs=self.mode_kwargs
                                        )
            if not quantum_instance:
                if not self.quantum_instance:
                    raise AquaError('QuantumInstance must be provided, or QKernelRegression must have already been run')
                quantum_instance = self.quantum_instance
            new_qkr.run(quantum_instance)
            predict, error = new_qkr.predict(x.reshape(1, -1), new_y_to_score=np.array(y).reshape(1))
            predictions += [predict]
            errors += [error]
        return predictions, errors