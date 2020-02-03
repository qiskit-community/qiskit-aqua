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
This module contains the definition of a base class for Oracle.
"""

from abc import ABC, abstractmethod


class Oracle(ABC):

    """
        Base class for oracles.

        This method should initialize the module and
        use an exception if a component of the module is not
        available.

        Args:
            args (list): args
            kwargs (dict): kwargs
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        self._output_register = None
        self._variable_register = None
        self._ancillary_register = None
        self._circuit = None

    @property
    def circuit(self):
        """ circuit """
        if self._circuit is None:
            self._circuit = self.construct_circuit()
        return self._circuit

    @property
    @abstractmethod
    def variable_register(self):
        """ returns variable register """
        raise NotImplementedError()

    @property
    @abstractmethod
    def ancillary_register(self):
        """ returns ancillary register """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_register(self):
        """ returns output register """
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self):
        """Construct the oracle circuit.

        Returns:
            A quantum circuit for the oracle.
        """
        raise NotImplementedError()
