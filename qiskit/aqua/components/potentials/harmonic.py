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
This module contains the definition of a base class for potentials.
"""


from abc import abstractmethod

from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import Pluggable, AquaError
import numpy as np
import scipy.linalg as lng

from . import Potential

class Harmonic(Potential):

    """Base class for iHarmonic Potentials: 1/2 m w^2 (x0+delta*x)^2

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    #@abstractmethod
    def __init__(self, num_qubits, const, x0, delta, tau):
        super().__init__()
        self._num_qubits = num_qubits
        self._c = const
        self._x0 = x0
        self._delta = delta
        self._tau = tau

    # @classmethod
    # def init_params(cls, num_qubits, m, omega, x0, delta):
    #     cls._num_qubits = num_qubits
    #     cls._m = m
    #     cls._omega = omega
    #     cls._x0 = x0
    #     cls._delta = delta

    #@abstractmethod
    def construct_circuit(self, mode, ordering = 'normal', register=None):
        """
        Construct a circuit to apply a harmonic potential on the statevector.

        Args:

        Returns:

        Raises:
        """

        if mode=='matrix':

            circ = np.zeros((1<<self._num_qubits,1<<self._num_qubits), dtype='complex64')
            if ordering == 'normal':
                for i in range(1<<self._num_qubits):
                    circ[i,i]=-1.j * 0.5 * self._c * (self._x0 + i*self._delta)**2 * self._tau

            elif ordering == 'reversed':
                for i in range(1<<self._num_qubits):
                    j = (1<<self._num_qubits)-1-i
                    circ[j,j]=-1.j * 0.5 * self._c * (self._x0 + i*self._delta)**2 * self._tau

            else:
                raise ValueError('Ordering should be either "normal" or "reversed"')


            return lng.expm(circ)


        elif mode=='circuit':

            if ordering == 'normal':
                gamma = 0.5 * self._c *self._tau

                q = QuantumRegister(self._num_qubits, name='q')
                circ = QuantumCircuit(q)

                #global phase
                circ.u1(-1 * gamma * self._x0**2, q[0])
                circ.x(q[0])
                circ.u1(-1 * gamma * self._x0**2, q[0])
                circ.x(q[0])

                # phase shift
                for i in range(self._num_qubits):
                    circ.u1(-2 * gamma * self._x0 * self._delta * 2**i, q[i])

                #controlled phase shift
                for i in range(self._num_qubits):
                    for j in range(self._num_qubits):
                        if i == j:
                            circ.u1(-1 * gamma * self._delta**2 * 2**(i+j), q[i])
                        else:
                            circ.cu1(-1 * gamma * self._delta**2 * 2**i * 2**j, q[i], q[j])

            elif ordering == 'reversed':

                gamma = 0.5 * self._c *self._tau

                q = QuantumRegister(self._num_qubits, name='q')
                circ = QuantumCircuit(q)

                #global phase
                circ.u1(-1 * gamma * self._x0**2, q[0])
                circ.x(q[0])
                circ.u1(-1 * gamma * self._x0**2, q[0])
                circ.x(q[0])

                # phase shift
                for i in range(self._num_qubits):
                    circ.u1(-2 * gamma * self._x0 * self._delta * 2**(self._num_qubits-1-i), q[i])

                #controlled phase shift
                for i in range(self._num_qubits):
                    for j in range(self._num_qubits):
                        if i == j:
                            circ.u1(-1 * gamma * self._delta**2 * 2**(2*(self._num_qubits-1-i)), q[i])
                        else:
                            circ.cu1(-1 * gamma * self._delta**2 * 2**(self._num_qubits-1-i) * 2**(self._num_qubits-1-j), q[i], q[j])

            else:

                raise ValueError('Ordering should be either "normal" or "reversed"')


            return circ

