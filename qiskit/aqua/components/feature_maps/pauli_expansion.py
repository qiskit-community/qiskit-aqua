# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Pauli Expansion feature map."""

import warnings
from typing import Optional, Callable, List
import itertools
import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi

from qiskit.aqua.operators import evolution_instruction
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .feature_map import FeatureMap
from .data_mapping import self_product

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class PauliExpansion(FeatureMap):
    r"""DEPRECATED. The Pauli Expansion feature map.

    Refer to https://arxiv.org/abs/1804.11326 for details.

    The Pauli Expansion feature map transforms data :math:`\vec{x} \in \mathbb{R}^n`
    according to the following equation, and then duplicate the same circuit with depth
    :math:`d` times, where :math:`d` is the depth of the circuit:

    :math:`U_{\Phi(\vec{x})}=\exp\left(i\sum_{S\subseteq [n]}
    \phi_S(\vec{x})\prod_{i\in S} P_i\right)`

    where :math:`S \in \{\binom{n}{k}\ combinations,\ k = 1,... n \}, \phi_S(\vec{x}) = x_i` if
    :math:`k=1`, otherwise :math:`\phi_S(\vec{x}) = \prod_S(\pi - x_j)`, where :math:`j \in S`, and
    :math:`P_i \in \{ I, X, Y, Z \}`

    Please refer to :class:`FirstOrderExpansion` for the case
    :math:`k = 1`, :math:`P_0 = Z`
    and to :class:`SecondOrderExpansion` for the case
    :math:`k = 2`, :math:`P_0 = Z\ and\ P_1 P_0 = ZZ`.
    """

    def __init__(self,
                 feature_dimension: int,
                 depth: int = 2,
                 entangler_map: Optional[List[List[int]]] = None,
                 entanglement: str = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Callable[[np.ndarray], float] = self_product) -> None:
        """
        Args:
            feature_dimension: The number of features
            depth: The number of repeated circuits. Defaults to 2, has a minimum value of 1.
            entangler_map: Describes the connectivity of qubits, each list in the overall list
                describes [source, target]. Defaults to ``None`` where the map is created as per
                *entanglement* parameter.
                Note that the order in the list is the order of applying the two-qubit gate.
            entanglement: ('full' | 'linear'), generate the qubit connectivity by a predefined
                topology. Defaults to full which connects every qubit to each other. Linear
                connects each qubit to the next.
            paulis: a list of strings for to-be-used paulis (a pauli is a any combination
                of I, X, Y ,Z). Note that the order of pauli label is counted from
                right to left as the notation used in Pauli class in Qiskit Terra.
                Defaults to ``None`` whereupon ['Z', 'ZZ'] will be used.
            data_map_func: A mapping function for data x which can be supplied to override the
                default mapping from :meth:`self_product`.
        """
        warnings.warn('The qiskit.aqua.components.feature_maps.PauliExpansion object is '
                      'deprecated as of 0.7.0 and will be removed no sooner than 3 months after '
                      'the release. You should use qiskit.circuit.library.PauliFeatureMap instead.',
                      DeprecationWarning, stacklevel=2)

        paulis = paulis if paulis is not None else ['Z', 'ZZ']
        validate_min('depth', depth, 1)
        validate_in_set('entanglement', entanglement, {'full', 'linear'})
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
        """Construct the second order expansion based on given data.

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
