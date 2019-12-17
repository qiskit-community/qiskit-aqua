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
This module contains the definition of a base class for
initial states. An initial state might be used by a variational
form or in eoh as a trial state to evolve
"""

from abc import abstractmethod
from qiskit.circuit import QuantumRegister  # pylint: disable=unused-import
from qiskit.aqua import Pluggable, AquaError  # pylint: disable=unused-import


class InitialState(Pluggable):
    """Base class for InitialState.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @classmethod
    def init_params(cls, params):
        """ init params """
        init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
        args = {k: v for k, v in init_state_params.items() if k != 'name'}
        return cls(**args)

    @abstractmethod
    def construct_circuit(self, mode='circuit', register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): qubits for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            AquaError: when mode is not 'vector' or 'circuit'.
        """
        raise NotImplementedError()

    @property
    def bitstr(self):
        """ bitstr """
        return None
