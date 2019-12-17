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

"""A normal standard IQFT."""

from scipy import linalg

from .approximate import Approximate


class Standard(Approximate):
    """A normal standard IQFT."""

    CONFIGURATION = {
        'name': 'STANDARD',
        'description': 'Inverse QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'std_iqft_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits):
        super().__init__(num_qubits, degree=0)

    def _build_matrix(self):
        # pylint: disable=no-member
        return linalg.dft(2 ** self._num_qubits, scale='sqrtn')
