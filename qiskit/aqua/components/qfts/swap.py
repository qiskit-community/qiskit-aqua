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
import numpy as np

from qiskit.qasm import pi

# from . import QFT
# from .qft import set_up
#
#
# class Swap(QFT):
#     """A normal standard QFT."""
#
#     CONFIGURATION = {
#         'name': 'SWAP',
#         'description': 'QFT',
#         'input_schema': {
#             '$schema': 'http://json-schema.org/schema#',
#             'id': 'std_qft_schema',
#             'type': 'object',
#             'properties': {
#             },
#             'additionalProperties': False
#         }
#     }
#
#     def __init__(self, num_qubits):
#         super().__init__()
#         self._num_qubits = num_qubits
#
#     def construct_circuit(self, mode='circuit', qubits=None, circuit=None):
#         if mode == 'vector':
#             raise ValueError('Mode should be "circuit"')
#         elif mode == 'circuit':
#             circuit, qubits = set_up(circuit, qubits, self._num_qubits)
#
#             for i in range(int(self._num_qubits/2)):
#
#                 circuit.swap(qubits[i],qubits[self._num_qubits-1-i])
#
#             return circuit
#         else:
#             raise ValueError('Mode should be either "vector" or "circuit"')
