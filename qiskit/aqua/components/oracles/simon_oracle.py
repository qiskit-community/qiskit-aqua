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

import logging
import math
import numpy
from sympy import Matrix, mod_inverse

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


class SimonOracle(Oracle):

    CONFIGURATION = {
        'name': 'SimonOracle',
        'description': 'Simon Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'simon_oracle_schema',
            'type': 'object',
            'properties': {
                'bitmap': {
                    "type": ["object", "null"],
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, bitmap):
        self.validate(locals())
        super().__init__()

        # checks that the input bitstring length is a power of two
        nbits = math.log(len(bitmap), 2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AlgorithmError('Input not the right length')
        self._nbits = int(nbits)

        # find the two keys that have matching values
        get_key_pair = ((k1, k2) for k1, v1 in bitmap.items()
                        for k2, v2 in bitmap.items()
                        if v1 == v2 and not k1 == k2)
        try:  # matching keys found
            k1, k2 = next(get_key_pair)
            self._hidden = numpy.binary_repr(int(k1, 2) ^ int(k2, 2), nbits)
        except StopIteration as e:  # non matching keys found
            k1, k2 = None, None
            self._hidden = numpy.binary_repr(0, self._nbits)

        self._qr_variable = QuantumRegister(self._nbits, name='v')
        self._qr_ancilla = QuantumRegister(self._nbits, name='a')

    def variable_register(self):
        return self._qr_variable

    def ancillary_register(self):
        return self._qr_ancilla

    def outcome_register(self):
        pass

    def construct_circuit(self):
        qc = QuantumCircuit(self._qr_variable, self._qr_ancilla)

        # Copy the content of the variable register to the ancilla register
        for i in range(self._nbits):
            qc.cx(self._qr_variable[i], self._qr_ancilla[i])

        # Find the first occurance of 1 in the hidden string
        flip_index = self._hidden.find("1")

        # Create 1-to-1 or 2-to-1 mapping
        for i, c in enumerate(self._hidden):
            if c == "1" and flip_index > -1:
                qc.cx(self._qr_variable[flip_index], self._qr_ancilla[i])

        # Randomly permute the ancilla register
        perm = list(numpy.random.permutation(self._nbits))
        init = list(range(self._nbits))
        i = 0
        while i < self._nbits:
            if init[i] != perm[i]:
                k = perm.index(init[i])
                qc.swap(self._qr_ancilla[i], self._qr_ancilla[k])
                init[i], init[k] = init[k], init[i]
            else:
                i += 1

        # Randomly flip bits in the ancilla register
        for i in range(self._nbits):
            if numpy.random.random() > 0.5:
                qc.x(self._qr_ancilla[i])

        return qc

    def evaluate_classically(self, assignment):
        return self._hidden == assignment

    def interpret_measurement(self, measurement, *args, **kwargs):
        # reverse measurement bitstrings and remove all zero entry
        linear = [(k[::-1], v) for k, v in measurement.items()
                  if k != "0"*self._nbits]
        # sort bitstrings by their probailities
        linear.sort(key=lambda x: x[1], reverse=True)

        # construct matrix
        equations = []
        for k, v in linear:
            equations.append([int(c) for c in k])
        y = Matrix(equations)

        # perform gaussian elimination
        y_transformed = y.rref(iszerofunc=lambda x: x % 2 == 0)

        def mod(x, modulus):
            numer, denom = x.as_numer_denom()
            return numer*mod_inverse(denom, modulus) % modulus
        y_new = y_transformed[0].applyfunc(lambda x: mod(x, 2))

        # determine hidden string from final matrix
        rows, cols = y_new.shape
        hidden = [0]*self._nbits
        for r in range(rows):
            yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
            if len(yi) == 2:
                hidden[yi[0]] = '1'
                hidden[yi[1]] = '1'
        return "".join(str(x) for x in hidden)
