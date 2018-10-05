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
"""
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""


import numpy as np
from qiskit import CompositeGate, QuantumCircuit, QuantumRegister
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate

from qiskit_aqua.algorithms.components.feature_maps import FeatureMap


class SecondOrderExpansion(FeatureMap):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    SECOND_ORDER_EXPANSION_CONFIGURATION = {
        'name': 'SecondOrderExpansion',
        'description': 'Second order expansion for feature map',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Second_Order_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['object', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.SECOND_ORDER_EXPANSION_CONFIGURATION.copy())
        self._ret = {}

    def init_args(self, num_qubits, depth, entangler_map=None, entanglement='full'):
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = self.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = self.validate_entangler_map(entangler_map, num_qubits)

    def _build_composite_gate(self, x, qr):
        composite_gate = CompositeGate("second_order_expansion",
                                       [], [qr[i] for i in range(self._num_qubits)])

        for _ in range(self._depth):
            for i in range(x.shape[0]):
                composite_gate._attach(U2Gate(0, np.pi, qr[i]))
                composite_gate._attach(U1Gate(2 * x[i], qr[i]))
            for src, targs in self._entangler_map.items():
                for targ in targs:
                    composite_gate._attach(CnotGate(qr[src], qr[targ]))
                    # TODO, it might not need pi - x
                    composite_gate._attach(U1Gate(2 * (np.pi - x[src]) * (np.pi - x[targ]),
                                                  qr[targ]))
                    composite_gate._attach(CnotGate(qr[src], qr[targ]))

        return composite_gate

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be numpy array.")
        if x.ndim != 1:
            raise ValueError("x must be 1-D array.")
        if x.shape[0] != self._num_qubits:
            raise ValueError("number of qubits and data dimension must be the same.")

        if qr is None:
            qr = QuantumRegister(self._num_qubits, 'q')
        qc = QuantumCircuit(qr)
        composite_gate = self._build_composite_gate(x, qr)
        qc._attach(composite_gate if not inverse else composite_gate.inverse())

        return qc
