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
from qiskit.aqua import Pluggable
from abc import abstractmethod
from qiskit.aqua.utils import get_entangler_map, validate_entangler_map


class FeatureMap(Pluggable):

    """Base class for FeatureMap.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        feat_map__params = params.get(Pluggable.SECTION_KEY_FEATURE_MAP)
        args = {k: v for k, v in feat_map__params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def construct_circuit(self, x, qr=None, inverse=False):
        """Construct the variational form, given its parameters.

        Args:
            x (numpy.ndarray[float]): 1-D array, data
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit

        Returns:
            QuantumCircuit: a quantum circuit.
        """
        raise NotImplementedError()

    @staticmethod
    def get_entangler_map(map_type, num_qubits):
        return get_entangler_map(map_type, num_qubits)

    @staticmethod
    def validate_entangler_map(entangler_map, num_qubits):
        return validate_entangler_map(entangler_map, num_qubits)

    @property
    def feature_dimension(self):
        return self._feature_dimension

    @property
    def num_qubits(self):
        return self._num_qubits
