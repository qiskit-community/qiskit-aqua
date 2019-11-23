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
This module contains the definition of a base class for
feature map. Several types of commonly used approaches.
"""

import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi

from qiskit.aqua.operators import evolution_instruction
from qiskit.aqua.components.feature_maps import FeatureMap, self_product

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class PauliExpansion(FeatureMap):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'PauliExpansion',
        'description': 'Pauli expansion for feature map (any order)',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'Pauli_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'enum': ['full', 'linear']
                },
                'paulis': {
                    'type': ['array', 'null'],
                    'items': {
                        'type': 'string'
                    },
                    'default': None
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, feature_dimension, depth=2, entangler_map=None,
                 entanglement='full', paulis=None, data_map_func=self_product):
        """Constructor.

        Args:
            feature_dimension (int): number of features
            depth (Optional(int)): the number of repeated circuits. Defaults to 2
            entangler_map (Optional(list[list])): describe the connectivity of qubits,
                                        each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (Optional((str)): ['full', 'linear'], generate the qubit
                                          connectivity by predefined topology.
                                          Defaults to full
            paulis (Optional(list[str])): a list of strings for to-be-used paulis.
                                    Defaults to None. If None, ['Z', 'ZZ'] will be used.
            data_map_func (Optional(Callable)): a mapping function for data x
        """
        paulis = paulis if paulis is not None else ['Z', 'ZZ']
        self.validate(locals())
        super().__init__()
        self._num_qubits = self._feature_dimension = feature_dimension
        self._depth = depth
        if entangler_map is None:
            self._entangler_map = self.get_entangler_map(entanglement, feature_dimension)
        else:
            self._entangler_map = self.validate_entangler_map(entangler_map, feature_dimension)

        self._pauli_strings = self._build_subset_paulis_string(paulis)
        self._data_map_func = data_map_func
        self._support_parameterized_circuit = True

    def _build_subset_paulis_string(self, paulis):
        # fill out the paulis to the number of qubits
        temp_paulis = []
        for pauli in paulis:
            len_pauli = len(pauli)
            for possible_pauli_idx in itertools.combinations(range(self._num_qubits), len_pauli):
                string_temp = ['I'] * self._num_qubits
                for idx, _ in enumerate(possible_pauli_idx):
                    string_temp[-possible_pauli_idx[idx] - 1] = pauli[-idx - 1]
                temp_paulis.append(''.join(string_temp))
        # clean up string that can not be entangled.
        final_paulis = []
        for pauli in temp_paulis:
            where_z = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
            if len(where_z) == 1:
                final_paulis.append(pauli)
            else:
                is_valid = True
                for src, targ in itertools.combinations(where_z, 2):
                    if [src, targ] not in self._entangler_map:
                        is_valid = False
                        break
                if is_valid:
                    final_paulis.append(pauli)
                else:
                    logger.warning("Due to the limited entangler_map,"
                                   " %s is skipped.", pauli)
        logger.info("Pauli terms include: %s", final_paulis)
        return final_paulis

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != 'I')[0]
        x = np.asarray(x)
        return x[where_non_i]

    def construct_circuit(self, x, qr=None, inverse=False):
        """
        Construct the second order expansion based on given data.

        Args:
            x (Union(numpy.ndarray, list[Parameter], ParameterVector)): 1-D to-be-transformed data.
            qr (QuantumRegister, optional): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool, optional): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        Raises:
            TypeError: invalid input
            ValueError: invalid input
        """
        if len(x) != self._num_qubits:
            raise ValueError("number of qubits and data dimension must be the same.")

        if qr is None:
            qr = QuantumRegister(self._num_qubits, name='q')

        qc = QuantumCircuit(qr)
        for _ in range(self._depth):
            for i in range(self._num_qubits):
                qc.u2(0, pi, qr[i])
            for pauli in self._pauli_strings:
                coeff = self._data_map_func(self._extract_data_for_rotation(pauli, x))
                p = Pauli.from_label(pauli)
                inst = evolution_instruction([[1, p]], coeff, 1)
                qc.append(inst, qr)
        return qc
