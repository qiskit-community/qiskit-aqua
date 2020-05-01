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
The Univariate Uniform Distribution.
"""

import numpy as np
from .univariate_distribution import UnivariateDistribution


class UniformDistribution(UnivariateDistribution):
    """
    The Univariate Uniform Distribution.
    """

    CONFIGURATION = {
        'name': 'UniformDistribution',
        'description': 'Uniform Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'UniformDistribution_schema',
            'type': 'object',
            'properties': {
                'num_target_qubits': {
                    'type': 'integer',
                    'default': 2,
                },
                'low': {
                    'type': 'number',
                    'default': 0,
                },
                'high': {
                    'type': 'number',
                    'default': 1,
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_target_qubits, low=0, high=1):
        """
        Univariate uniform distribution
        Args:
            num_target_qubits (int): number of qubits it acts on
            low (float): lower bound, i.e., the value corresponding \
                        to |0...0> (assuming an equidistant grid)
            high (float): upper bound, i.e., the value corresponding \
                        to |1...1> (assuming an equidistant grid)
        """
        probabilities = np.ones(2**num_target_qubits)/2**num_target_qubits
        super().__init__(num_target_qubits, probabilities, low, high)

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            for i in range(self.num_target_qubits):
                qc.h(q[i])
        else:
            for i in params['i_state']:
                qc.h(q[i])
