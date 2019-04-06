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
This module contains the definition of a base class for inverse quantum fourier transforms.
"""

from abc import abstractmethod

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import Pluggable, AquaError


class IQFT(Pluggable):

    """Base class for Inverse QFT.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        iqft_params = params.get(Pluggable.SECTION_KEY_IQFT)
        kwargs = {k: v for k, v in iqft_params.items() if k != 'name'}
        return cls(**kwargs)

    @abstractmethod
    def _build_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        raise NotImplementedError

    def construct_circuit(self, mode='circuit', qubits=None, circuit=None, do_swaps=True):
        """Construct the circuit.

        Args:
            mode (str): 'matrix' or 'circuit'
            qubits (QuantumRegister or qubits): register or qubits to build the circuit on.
            circuit (QuantumCircuit): circuit for construction.
            do_swaps (bool): include the swaps.

        Returns:
            The matrix or circuit depending on the specified mode.
        """
        if mode == 'circuit':
            return self._build_circuit(qubits=qubits, circuit=circuit, do_swaps=do_swaps)
        elif mode == 'matrix':
            return self._build_matrix()
        else:
            raise AquaError('Unrecognized mode: {}.'.format(mode))
