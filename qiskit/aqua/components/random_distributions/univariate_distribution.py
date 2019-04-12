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
This module contains the definition of a base class for univariate distributions.
"""

import numpy as np
import sys, math
from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import Custom
from .random_distribution import RandomDistribution
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer, execute


class UnivariateDistribution(RandomDistribution):
    """
    This module contains the definition of a base class for univariate distributions.
    (Interface for discrete bounded uncertainty models assuming an equidistant grid)
    """

    def __init__(self, num_target_qubits, probabilities, low: float=0, high: float=1, backend=None):
        super().__init__(num_target_qubits)
        self._num_values = 2 ** self.num_target_qubits
        self._probabilities = np.array(probabilities)
        self._low = low
        self._high = high
        self._values = np.linspace(low, high, self.num_values)
        if self.num_values != len(probabilities):
            raise AquaError('num qubits and length of probabilities vector do not match!')

        #####################
        # XXX Albert's stuff.
        # TODO: where to define backend? By default it should be IBMQ.
        assert isinstance(num_target_qubits, int) and num_target_qubits > 0
        q = QuantumRegister(num_target_qubits)
        c = ClassicalRegister(num_target_qubits)
        self._circuit = QuantumCircuit(q, c)
        self._circuit.h(q)
        self._circuit.barrier()
        self._circuit.measure(q, c)
        self._backend = backend if backend != None else BasicAer.get_backend('qasm_simulator')

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    @property
    def num_values(self):
        return self._num_values

    @property
    def values(self):
        return self._values

    @property
    def probabilities(self):
        return self._probabilities

    def required_ancillas(self):
        return 0

    def required_ancillas_controlled(self):
        return 0

    def build(self, qc, q, q_ancillas=None, params=None):
        custom_state = Custom(self.num_target_qubits, state_vector=np.sqrt(self.probabilities))
        qc.extend(custom_state.construct_circuit('circuit', q))

    @staticmethod
    def pdf_to_probabilities(pdf, low, high, num_values):
        probabilities = np.zeros(num_values)
        values = np.linspace(low, high, num_values)
        total = 0
        for i, x in enumerate(values):
            probabilities[i] = pdf(values[i])
            total += probabilities[i]
        probabilities /= total
        return probabilities, values

    #####################
    # XXX Albert's stuff.

    def uniform_rand_float64(self, size: int, vmin: float, vmax: float) -> np.ndarray:
        """
        Generates a vector of random float64 values in the range [vmin, vmax].
        :param size: length of the vector.
        :param vmin: lower bound.
        :param vmax: upper bound.
        :return: vector of random values.
        """
        assert sys.maxsize == np.iinfo(np.int64).max                                # sizeof(int) == 64 bits
        assert isinstance(size, int) and size > 0
        assert isinstance(vmin, float) and isinstance(vmax, float) and vmin <= vmax
        nbits = 7 * 8                                                               # nbits > mantissa of float64
        bit_str_len = (nbits * size + self.num_target_qubits - 1) // self.num_target_qubits
        job = execute(self._circuit, self._backend, shots=bit_str_len, memory=True)
        bit_str = ''.join(job.result().get_memory())
        scale = float(vmax - vmin) / float(2**nbits - 1)
        return np.array([vmin + scale * float(int(bit_str[i:i+nbits], 2))
                         for i in range(0, nbits * size, nbits)], dtype=np.float64)

    def uniform_rand_int64(self, size: int, vmin: int, vmax: int) -> np.ndarray:
        """
        Generates a vector of random int64 values in the range [vmin, vmax].
        :param size: length of the vector.
        :param vmin: lower bound.
        :param vmax: upper bound.
        :return: vector of random values.
        """
        assert sys.maxsize == np.iinfo(np.int64).max                                # sizeof(int) == 64 bits
        assert isinstance(size, int) and size > 0
        assert isinstance(vmin, int) and isinstance(vmax, int) and vmin <= vmax
        assert abs(vmin) <= 2**52 and abs(vmax) <= 2**52                            # 52 == mantissa of float64
        return np.rint(self.uniform_rand_float64(size, float(vmin), float(vmax))).astype(np.int64)
