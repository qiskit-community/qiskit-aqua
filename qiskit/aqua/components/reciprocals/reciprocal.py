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
This module contains the definition of a base class for computing reciprocals
into an amplitude.
"""

from abc import abstractmethod
from qiskit.aqua import Pluggable


class Reciprocal(Pluggable):

    """Base class for reciprocal calculation.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        """ init params """
        reci_params = params.get(Pluggable.SECTION_KEY_RECIPROCAL)
        args = {k: v for k, v in reci_params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def sv_to_resvec(self, statevector, num_q):
        """Convert statevector to result vector.

        Args:
            statevector (list): The statevector from simulation.
            num_q (int): Number of qubits of result register.

        Returns:
             The result vector.
        """
        raise NotImplementedError

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
