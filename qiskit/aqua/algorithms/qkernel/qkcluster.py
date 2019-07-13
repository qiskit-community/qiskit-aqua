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
Quantum Kernel Clustering algorithm.

Anna Phan & Donny Greenberg (contributed equally)
Publication pending.

"""

import logging
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import davies_bouldin_score

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, QuantumAlgorithm, register_pluggable, AquaError

from qiskit.aqua.algorithms.qkernel import QKernel

logger = logging.getLogger(__name__)


class QKernelCluster(QuantumAlgorithm):
    CONFIGURATION = {
        'name': 'QKernel.Cluster',
        'description': 'QKernel Clustering Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKernel_cluster_schema',
            'type': 'object',
            'properties': {
                'num_clusters': {
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
        'problems': ['clustering'],
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

    def __init__(self, circuit_maker, dataset, num_clusters, modes='all', qkernel=None):
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

        all_modes = ['spectral', 'dbscan', 'agglomerative']
        if modes == 'all':
            self.modes = all_modes
        elif set(modes) <= set(all_modes):
            self.modes = modes
        else:
            raise ValueError('Clustering mode {} not supported'.format(modes))

        if dataset is None:
            raise ValueError('Dataset must be provided')
        self.dataset = dataset
        self.num_clusters = num_clusters

        self._return = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        dataset = algo_input

        if dataset is None:
            raise ValueError('Dataset must be provided')

        if not isinstance(dataset, np.ndarray):
            dataset = np.asarray(dataset)

        cluster_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_clusters = cluster_params.get('num_clusters')
        num_qubits = cluster_params.get('num_qubits')

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

        return cls(circuit_maker, dataset, num_clusters)

    def _run(self):
        if not self.qkernel.kernel_matrix:
            self.qkernel.construct_kernel_matrix(self.dataset, quantum_instance=self.quantum_instance,preserve_counts=True)

        overlap = self.qkernel.kernel_matrix
        self._return['overlap_matrix'] = overlap
        
        distance = self.qkernel.distance_matrix
        self._return['distance_matrix'] = distance

        for mode in self.modes:
            model = None
            if mode == 'spectral':
                model = SpectralClustering(n_clusters=self.num_clusters, affinity='precomputed')
                model.fit_predict(overlap)
            elif mode == 'dbscan':
                model = DBSCAN(metric='precomputed', min_samples=2)
                model.fit(distance)
            elif mode == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='precomputed', linkage='average')
                model.fit(distance)
            else:
                raise ValueError('Unknown Clustering mode {} not supported'.format(mode))

            self._return[mode] = {}
            self._return[mode]['labels'] = model.labels_
            if np.min(model.labels_) != np.max(model.labels_):
                self._return[mode]['silhouette_score'] = silhouette_score(distance, model.labels_, metric="precomputed")
                self._return[mode]['calinski_harabaz_score'] = calinski_harabaz_score(self.dataset, model.labels_)
                self._return[mode]['davies_bouldin_score'] = davies_bouldin_score(self.dataset, model.labels_)
            # TODO too heavy to return the whole model?
            # self._return[mode]['model'] = model

        return self._return