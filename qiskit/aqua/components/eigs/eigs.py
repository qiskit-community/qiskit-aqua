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
This module contains the definition of a base class for eigenvalue estimators.
"""
from qiskit.aqua import Pluggable
from qiskit import QuantumCircuit
from abc import abstractmethod


class Eigenvalues(Pluggable):
    """Base class for eigenvalue estimation.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    Args:
        params (dict): configuration dictionary
    """
    
    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        eigs_params = params.get(Pluggable.SECTION_KEY_EIGS)
        args = {k: v for k, v in eigs_params.items() if k != 'name'}
        return cls(**args)
    
    @abstractmethod
    def get_register_sizes(self):
        raise NotImplementedError()

    @abstractmethod
    def get_scaling(self):
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self, mode, register=None):
        """Construct the eigenvalue estimation quantum circuit.

        Args:
            mode (str): 'matrix' or 'circuit'
            register (QuantumRegister): register for circuit construction
                        where eigenvalues will be stored.

        Returns:
            QuantumCircuit object for the eigenvalue estimation circuit.
        """
        raise NotImplementedError()

    def construct_inverse(self, mode, circuit, inreg, outreg):
        """Construct the inverse eigenvalue estimation quantum circuit.

        Args:
            mode (str): consctruction mode, 'matrix' not supported
            circuit (QuantumCircuit): the quantum circuit to invert
            inreg (QuantumRegister): the input quantum register
            outreg (QuantumRegister): the output quantum register

        Returns:
            QuantumCircuit object for of the inverted eigenvalue estimation
            circuit.
        """
        if mode == 'matrix':
            raise NotImplementedError('The matrix mode is not supported.')
        if circuit is None:
            raise ValueError('Circuit was not constructed beforehand.')
        # qc = QuantumCircuit(inreg, outreg)
        # for gate in reversed(circuit.data):
        #     gate.reapply(qc)
        #     qc.data[-1].inverse()
        self._inverse = circuit.inverse()
        return self._inverse
