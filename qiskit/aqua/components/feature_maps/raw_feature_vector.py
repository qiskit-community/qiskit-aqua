# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

import logging

import numpy as np
from qiskit import QuantumCircuit  # pylint: disable=unused-import

from qiskit.aqua.utils.arithmetic import next_power_of_2_base
from qiskit.aqua.components.feature_maps import FeatureMap
from qiskit.aqua.circuits import StateVectorCircuit

logger = logging.getLogger(__name__)


class RawFeatureVector(FeatureMap):
    """
    Using raw feature vector as the initial state vector
    """

    CONFIGURATION = {
        'name': 'RawFeatureVector',
        'description': 'Raw feature vector',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'raw_feature_vector_schema',
            'type': 'object',
            'properties': {
                'feature_dimension': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, feature_dimension=2):
        """Constructor.

        Args:
            feature_dimension (int): The feature dimension
        """
        self.validate(locals())
        super().__init__()
        self._feature_dimension = feature_dimension
        self._num_qubits = next_power_of_2_base(feature_dimension)

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (numpy.ndarray): 1-D to-be-encoded data.
            qr (QuantumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): inverse
        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        Raises:
            TypeError: invalid input
            ValueError: invalid input
        """
        if len(x) != self._feature_dimension:
            raise ValueError("Unexpected feature vector dimension.")

        state_vector = np.pad(x, (0, (1 << self.num_qubits) - len(x)), 'constant')

        svc = StateVectorCircuit(state_vector)
        return svc.construct_circuit(register=qr)
