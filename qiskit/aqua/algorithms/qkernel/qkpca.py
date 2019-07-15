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
Quantum Kernel PCA algorithm.

Donny Greenberg & Anna Phan (contributed equally)
Publication pending.

"""

import logging
import numpy as np

from sklearn.decomposition import KernelPCA

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, QuantumAlgorithm, AquaError

from qiskit.aqua.algorithms.qkernel import QKernel

logger = logging.getLogger(__name__)


class QKernelPCA(QuantumAlgorithm):
    CONFIGURATION = {
        'name': 'QKernel.PCA',
        'description': 'QKernel Principal Component Analysis Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKernel_pca_schema',
            'type': 'object',
            'properties': {
                'num_components': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'num_qubits': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['dimensionality_reduction'],
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

    def __init__(self, circuit_maker, dataset, num_components=2, qkernel=None):
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

        if dataset is None:
            raise ValueError('Dataset must be provided')
        self.dataset = dataset
        self.num_components = num_components

        self._pca = None
        self._return = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        dataset = algo_input

        if dataset is None:
            raise ValueError('Dataset must be provided')

        if not isinstance(dataset, np.ndarray):
            dataset = np.asarray(dataset)

        pca_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_components = pca_params.get('num_components')
        num_qubits = pca_params.get('num_qubits')

        # Default circuit maker is feature map. If none present, check for variational form.
        if dataset.shape[1] == num_qubits and 'feature_map' in params:
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

        return cls(circuit_maker, dataset, num_components)

    def _run(self):
        if not self.qkernel.kernel_matrix:
            self.qkernel.construct_kernel_matrix(self.dataset, quantum_instance=self.quantum_instance)
        kernel = self.qkernel.kernel_matrix
        self._return['overlap_matrix'] = kernel

        pca = KernelPCA(n_components=self.num_components, kernel='precomputed')
        kpca_transform = pca.fit_transform(kernel)
        self._pca = pca

        self._return['transformed_vectors'] = kpca_transform
        explained_variances = np.var(kpca_transform, axis=0)
        self._return['explained_variances'] = explained_variances
        explained_variance_ratios = explained_variances / np.nansum(explained_variances)
        self._return['explained_variance_ratios'] = explained_variance_ratios

        return self._return
