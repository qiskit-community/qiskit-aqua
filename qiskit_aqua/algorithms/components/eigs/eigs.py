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
from abc import ABC, abstractmethod

from qiskit import QuantumCircuit

class Eigenvalues(ABC):

    """Base class for Eigenvalues.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """
    
    @abstractmethod
    def __init__(self, configuration=None):
        self._configuration = configuration
        self._negative_evals = False
        self._output_register = None
        self._input_register = None
        self._circuit = None
        self._inverse = None
    
    @property
    def configuration(self):
        """Return configuration"""
        return self._configuration

    def init_params(self, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        self.init_args(**args)

    @abstractmethod
    def init_args(self, **args):
        raise NotImplementedError()
    
    @abstractmethod
    def get_register_sizes(self):
        raise NotImplementedError()

    @abstractmethod
    def get_scaling(self):
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self, mode, register=None):
        """Construct the eigenvalue estmiation circuit.

        Args:
            mode (str): 'vector' or 'circuit'
            eigs_register (QuantumRegister): register for circuit construction
                        where eigenvalues will be stored.
            computation_register (QuantumRegister): register for circuit construction
                        where inputvector is prepared.
            circuit (QuantumCircuit): circuit for construction.

        Returns:
            The iqft circuit.
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
            


