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

import itertools
import functools
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.tools.qi.pauli import label_to_pauli

from qiskit_aqua import Operator
from qiskit_aqua.algorithms.components.feature_maps import FeatureMap

logger = logging.getLogger(__name__)


class PauliExpansion(FeatureMap):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    PAULI_EXPANSION_CONFIGURATION = {
        'name': 'PauliZExpansion',
        'description': 'Pauli expansion for feature map (any order)',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Pauli_Expansion_schema',
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
                },
                'paulis': {
                    'type': 'string',
                    'default': 'Z'
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.PAULI_EXPANSION_CONFIGURATION.copy())
        self._ret = {}

    def init_args(self, num_qubits, depth, entangler_map=None, entanglement='full', paulis='Z'):
        """Initializer.

        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (dict):
            entanglement (str): ['full', 'linear']
            paulis (str): a comma-seperated string for to-be-used paulis
        """
        self._num_qubits = num_qubits
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = self.get_entangler_map(entanglement, num_qubits)
        else:
            self._entangler_map = self.validate_entangler_map(entangler_map, num_qubits)

        self._pauli_strings = self._build_subset_paulis_string(paulis)

    def _build_subset_paulis_string(self, paulis):

        all_paulis = paulis.strip().split(",")
        # fill out the paulis to the number of qubits
        temp_paulis = []
        for pauli in all_paulis:
            len_pauli = len(pauli)
            for j in itertools.combinations(range(self._num_qubits), len_pauli):
                string_temp = ['I'] * self._num_qubits
                for idx in range(len(j)):
                    string_temp[j[idx]] = pauli[idx]
                temp_paulis.append(''.join(string_temp))

        # clean up string that can not be entangled.
        final_paulis = []
        for pauli in temp_paulis:
            where_z = np.where(np.asarray(list(pauli)) != 'I')[0]
            if len(where_z) == 1:
                final_paulis.append(pauli)
            else:
                is_valid = True
                for src, targ in itertools.combinations(where_z, 2):
                    if src not in self._entangler_map:
                        is_valid = False
                        break
                    else:
                        if targ not in self._entangler_map[src]:
                            is_valid = False
                            break
                if is_valid:
                    final_paulis.append(pauli)
                else:
                    logger.warning("Due to the limited entangler_map, {} is skipped.".format(pauli))
        logger.info("Pauli terms include: {}".format(final_paulis))
        return final_paulis

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

        def get_coeff(paulis):
            # coeff for the rotation angle
            where_non_i = np.where(np.asarray(list(paulis)) != 'I')[0]
            coeff = x[where_non_i][0] if len(where_non_i) == 1 else \
                functools.reduce(lambda m, n: (np.pi - m) * (np.pi - n), x[where_non_i])
            return coeff

        if qr is None:
            qr = QuantumRegister(self._num_qubits)

        qc = QuantumCircuit(qr)
        for i in range(self._num_qubits):
            qc.u2(0, np.pi, qr[i])
        for pauli in self._pauli_strings:
            coeff = get_coeff(pauli)
            p = label_to_pauli(pauli)
            qc += Operator.construct_evolution_circuit([[coeff, p]], 1, 1, qr)

        qc.data *= self._depth
        if inverse:
            qc.data = [gate.inverse() for gate in reversed(qc.data)]
        return qc
