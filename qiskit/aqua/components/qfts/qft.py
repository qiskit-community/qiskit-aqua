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
This module contains the definition of a base class for quantum
fourier transforms.
"""
from abc import abstractmethod

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import Pluggable, AquaError


def set_up(circ, qubits, num_qubits):
    if circ:
        if not qubits:
            raise AquaError(
                'A QuantumRegister or a list of qubits need to be specified with the input QuantumCircuit.'
            )
    else:
        circ = QuantumCircuit()
        if not qubits:
            qubits = QuantumRegister(num_qubits, name='q')

    if len(qubits) < num_qubits:
        raise AquaError('Insufficient input qubits: {} provided but {} needed.'.format(
            len(qubits), num_qubits
        ))

    if isinstance(qubits, QuantumRegister):
        _ = qubits
    elif isinstance(qubits, list) and isinstance(qubits[0], tuple) and isinstance(qubits[0][0], QuantumRegister):
        _ = qubits[0][0]
    else:
        raise AquaError('Unrecognized input. Register or qubits expected.')
    if not circ.has_register(_):
        circ.add_register(_)
    return circ, qubits


class QFT(Pluggable):

    """Base class for QFT.

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
        qft_params = params.get(Pluggable.SECTION_KEY_QFT)
        kwargs = {k: v for k, v in qft_params.items() if k != 'name'}
        return cls(**kwargs)

    @abstractmethod
    def construct_circuit(self, mode, qubits=None, circuit=None):
        """Construct the qft circuit.

        Args:
            mode (str): 'vector' or 'circuit'
            qubits (QuantumRegister or qubits): register or qubits to build the qft circuit on.
            circuit (QuantumCircuit): circuit for construction.

        Returns:
            The qft circuit.
        """
        raise NotImplementedError()
