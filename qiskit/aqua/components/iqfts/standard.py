# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from scipy import linalg

from .approximate import Approximate


class Standard(Approximate):
    """A normal standard IQFT."""

    CONFIGURATION = {
        'name': 'STANDARD',
        'description': 'Inverse QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
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
        return linalg.dft(2 ** self._num_qubits, scale='sqrtn')
