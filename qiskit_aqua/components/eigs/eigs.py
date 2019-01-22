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
from qiskit_aqua import Pluggable
from abc import abstractmethod

from qiskit import QuantumCircuit


class Eigenvalues(Pluggable):
    """
    Base class for eigenvalue estimation.
    """
    
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._negative_evals = False
        self._output_register = None
        self._input_register = None
        self._circuit = None
        self._inverse = None

    @classmethod
    def init_params(cls, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        return cls(**args)
    
    @abstractmethod
    def get_register_sizes(self):
        raise NotImplementedError()

    @abstractmethod
    def get_scaling(self):
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self, mode, register=None):
        """
        Construct the eigenvalue estimation quantum circuit.
        Args:
            mode (str): 'vector' or 'circuit'
            register (QuantumRegister): register for circuit construction
                        where eigenvalues will be stored.
        Returns:
            the QuantumCircuit object for the eigenvalue estimation circuit.
        """
        raise NotImplementedError()

    def construct_inverse(self, mode):
        """ Construct the inverse to construct_circuit """
        if mode == "vector":
            raise NotImplementedError("Mode vector not supported by"
                    "construct_inverse.")
        if self._inverse is None:
            if self._circuit is None:
                raise ValueError("Circuit was not constructed beforehand.")
            qc = QuantumCircuit(self._input_register, self._output_register)
            for gate in reversed(self._circuit.data):
                gate.reapply(qc)
                qc.data[-1].inverse()
            self._inverse = qc
        return self._inverse
            


