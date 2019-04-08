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
This module contains the definition of a base class for computing reciprocals
into an amplitude.
"""
from qiskit.aqua import Pluggable
from abc import abstractmethod


class Reciprocal(Pluggable):

    """Base class for reciprocal calculation.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    Args:
        configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        reci_params = params.get(Pluggable.SECTION_KEY_RECIPROCAL)
        args = {k: v for k, v in reci_params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def construct_circuit(self, mode, register=None, circuit=None):
        """Construct the initial state circuit.

        Args:
            mode (str): 'matrix' or 'circuit'
            register (QuantumRegister): register for circuit construction.
            circuit (QuantumCircuit): circuit for construction.

        Returns:
            The reciprocal circuit.
        """
        raise NotImplementedError()
