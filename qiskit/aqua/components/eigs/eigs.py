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
This module contains the definition of a base class for eigenvalue estimators.
"""

from abc import abstractmethod
from qiskit.aqua import Pluggable


class Eigenvalues(Pluggable):
    """Base class for eigenvalue estimation.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()
        self._inverse = None

    @classmethod
    def init_params(cls, params):
        """ init params """
        eigs_params = params.get(Pluggable.SECTION_KEY_EIGS)
        args = {k: v for k, v in eigs_params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def get_register_sizes(self):
        """ get register sizes """
        raise NotImplementedError()

    @abstractmethod
    def get_scaling(self):
        """ get scaling """
        raise NotImplementedError()

    @abstractmethod
    def construct_circuit(self, mode, register=None):
        """Construct the eigenvalue estimation quantum circuit.

        Args:
            mode (str): 'matrix' or 'circuit'
            register (QuantumRegister): register for circuit construction
                        where eigenvalues will be stored.

        Returns:
            QuantumCircuit: object for the eigenvalue estimation circuit.
        Raises:
            NotImplementedError: not implemented
        """
        raise NotImplementedError()

    def construct_inverse(self, mode, circuit):
        """Construct the inverse eigenvalue estimation quantum circuit.

        Args:
            mode (str): construction mode, 'matrix' not supported
            circuit (QuantumCircuit): the quantum circuit to invert

        Returns:
            QuantumCircuit: object for of the inverted eigenvalue estimation
                            circuit.
        Raises:
            NotImplementedError: not implemented for matrix mode
            ValueError: Circuit was not constructed beforehand
        """
        if mode == 'matrix':
            raise NotImplementedError('The matrix mode is not supported.')
        if circuit is None:
            raise ValueError('Circuit was not constructed beforehand.')
        # TODO: need to check if circuit is empty, now, it enforce to put a barrier in evolution
        # instruction
        self._inverse = circuit.inverse()
        return self._inverse
