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
The Multivariate Uniform Distribution.
"""

import numpy as np
from qiskit.aqua.components.uncertainty_models.multivariate_distribution \
    import MultivariateDistribution


class MultivariateUniformDistribution(MultivariateDistribution):
    """
    The Multivariate Uniform Distribution.
    """

    CONFIGURATION = {
        'name': 'MultivariateUniformDistribution',
        'description': 'Multivariate Uniform Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'MultivariateUniformDistribution_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': 'array',
                    "items": {
                        "type": "number"
                    },
                    'default': [2, 2]
                },
                'low': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'high': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, low=None, high=None):
        """
        Multivariate uniform distribution
        Args:
            num_qubits (Union(list, numpy.ndarray)): list with the number of qubits per dimension
            low (Union(list, numpy.ndarray)): list with the lower bounds per dimension,
                                    set to 0 for each dimension if None
            high (Union(list, numpy.ndarray)): list with the upper bounds per dimension,
                                    set to 1 for each dimension if None
        """
        super().validate(locals())

        if low is None:
            low = np.zeros(num_qubits)
        if high is None:
            high = np.ones(num_qubits)

        num_values = np.prod([2**n for n in num_qubits])
        probabilities = np.ones(num_values)
        super().__init__(num_qubits, probabilities, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        if params is None or params['i_state'] is None:
            for i in range(sum(self.num_qubits)):
                qc.h(q[i])
        else:
            for qubits in params['i_state']:
                for i in qubits:
                    qc.h(q[i])
