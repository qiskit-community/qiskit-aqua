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
        self._N = 1<<num_qubits
        self._c = const
        self._x0 = x0
        self._delta = delta
        self._tau = tau

    #@abstractmethod
    def construct_circuit(self, mode, ordering = 'normal', shift = False, register=None):
        """
        Construct a circuit to apply a harmonic potential on the statevector.

        Args:

        Returns:

        Raises:
        """

        if mode=='matrix':

            circ = np.zeros((self._N,self._N), dtype='complex64')
            if ordering == 'normal':
                for i in range(self._N):
                    circ[i,i]=-1.j * 0.5 * self._c * (self._x0 + i*self._delta)**2 * self._tau

            elif ordering == 'reversed':
                for i in range(self._N):
                    bin_i = np.fromstring(np.binary_repr(i,width=self._num_qubits), dtype='S1').astype(int)
                    for k in range(int(self._num_qubits/2)):
                        i1 = bin_i[k]
                        i2 = bin_i[self._num_qubits-1-k]
                        bin_i[self._num_qubits-1-k]=i1
                        bin_i[k]=i2
                    j = 0
                    for k in range(self._num_qubits):
                        ki = int(bin_i[self._num_qubits-1-k])
                        j+= 2**k * ki

                    circ[j,j]=-1.j * 0.5 * self._c * (self._x0 + i*self._delta)**2 * self._tau


            else:
                raise ValueError('Ordering should be either "normal" or "reversed"')

            if shift:

                Id = np.identity(int(self._N / 2))
                Null = np.zeros((int(self._N / 2), int(self._N / 2)))
                A = np.concatenate((Null, Id), axis=1)
                B = np.concatenate((Id, Null), axis=1)
                X = np.concatenate((A, B), axis=0)

                circ = np.matmul(X, np.matmul(circ, X))

            return lng.expm(circ)


        elif mode=='circuit':

            if ordering == 'normal':
                gamma = 0.5 * self._c *self._tau

                q = QuantumRegister(self._num_qubits, name='q')
                circ = QuantumCircuit(q)

                if shift:
                    circ.x(q[self._num_qubits-1])

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

                if shift:
                    circ.x(q[self._num_qubits-1])

            elif ordering == 'reversed':

                gamma = 0.5 * self._c *self._tau

                q = QuantumRegister(self._num_qubits, name='q')
                circ = QuantumCircuit(q)

                if shift:
                    circ.x(q[self._num_qubits-1])

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

                if shift:
                    circ.x(q[self._num_qubits-1])

            else:

                raise ValueError('Ordering should be either "normal" or "reversed"')

            return circ



