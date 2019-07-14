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
Quantum Kernel Anomaly Detection algorithm, publication pending.

Anna Phan & Donny Greenberg (contributed equally)
Publication pending.

"""

import logging
import numpy as np

from sklearn.svm import OneClassSVM

from qiskit.aqua import Pluggable, PluggableType, get_pluggable_class, QuantumAlgorithm, AquaError

from qiskit.aqua.algorithms.qkernel import QKernel

logger = logging.getLogger(__name__)


class QKernelAnomalyDetection(QuantumAlgorithm):
    CONFIGURATION = {
        'name': 'QKernel.AnomalyDetection',
        'description': 'QKernel Anomaly Detection',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKernel_anomaly_detection_schema',
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
        'problems': ['anomaly_detection'],
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

    def __init__(self, circuit_maker, dataset, qkernel=None):
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

        self._model = None
        self._return = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        dataset = algo_input

        if dataset is None:
            raise ValueError('Dataset must be provided')

        if not isinstance(dataset, np.ndarray):
            dataset = np.asarray(dataset)

        model_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        num_qubits = model_params.get('num_qubits')

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

        return cls(circuit_maker, dataset)

    def _run(self):
        if not self.qkernel.kernel_matrix:
            self.qkernel.construct_kernel_matrix(self.dataset, quantum_instance=self.quantum_instance)
        kernel = self.qkernel.kernel_matrix
        self._return['overlap_matrix'] = kernel

        self._model = OneClassSVM(kernel='precomputed')
        labels = self._model.fit_predict(kernel)

        self._return['in_out_labels'] = labels

        # TODO success metrics? Predict new points?

        return self._return
