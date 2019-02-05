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
import operator

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


class BernsteinVaziraniOracle(Oracle):

    CONFIGURATION = {
        'name': 'BernsteinVaziraniOracle',
        'description': 'Bernstein Vazirani Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'bv_oracle_schema',
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

        # figure out the hidden parameter
        self._parameter = ""
        for i in range(self._nbits-1, -1, -1):
            bitstring = numpy.binary_repr(2**i, self._nbits)
            bit = bitmap[bitstring]
            self._parameter += bit

        self._bitmap = bitmap

        self._qr_variable = QuantumRegister(self._nbits, name='v')
        self._qr_ancilla = QuantumRegister(1, name='a')

    def variable_register(self):
        return self._qr_variable

    def ancillary_register(self):
        return self._qr_ancilla

    def outcome_register(self):
        pass

    def construct_circuit(self):
        qc = QuantumCircuit(self._qr_variable, self._qr_ancilla)

        for i in range(self._nbits):
            if (int(self._parameter) & (1 << i)):
                qc.cx(self._qr_variable[i], self._qr_ancilla[0])

        return qc

    def evaluate_classically(self, assignment):
        return self._parameter == assignment

    def interpret_measurement(self, measurement, *args, **kwargs):
        return max(measurement.items(), key=operator.itemgetter(1))[0]
