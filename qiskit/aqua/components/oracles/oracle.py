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
This module contains the definition of a base class for Oracle.
"""

from abc import abstractmethod

from qiskit.aqua import Pluggable


class Oracle(Pluggable):

    """
        Base class for oracles.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._output_register = None
        self._variable_register = None
        self._ancillary_register = None
        self._circuit = None

    @classmethod
    def init_params(cls, params):
        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        args = {k: v for k, v in oracle_params.items() if k != 'name'}
        return cls(**args)

    @property
    def circuit(self):
        if self._circuit is None:
            self._circuit = self.construct_circuit()
        return self._circuit

    @property
    @abstractmethod
    def variable_register(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def ancillary_register(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_register(self):
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self):
        """Construct the oracle circuit.

        Returns:
            A quantum circuit for the oracle.
        """
        raise NotImplementedError()
