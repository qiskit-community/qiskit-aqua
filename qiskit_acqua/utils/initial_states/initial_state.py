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
This module contains the definition of a base class for
initial states. An initial state might be used by a variational
form or in dynamics as a trial state to evolve
"""
from abc import ABC, abstractmethod


class InitialState(ABC):

    """Base class for InitialState.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    @abstractmethod
    def __init__(self, configuration=None):
        self._configuration = configuration

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
    def construct_circuit(self, mode, register=None):
        """Construct the initial state circuit.

        Args:
            mode (str): 'vector' or 'circuit'
            register (QuantumRegister): register for circuit construction.

        Returns:
            A quantum circuit.
        """
        raise NotImplementedError()

    @property
    def bitstr(self):
        return None
