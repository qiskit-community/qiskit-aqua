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
from qiskit_aqua.components.oracles import Oracle
from qiskit import QuantumRegister, QuantumCircuit 
import math
import numpy
import operator

logger = logging.getLogger(__name__)

class DeutschJozsaOracle(Oracle):

    CONFIGURATION = {
        'name': 'DeutschJozsaOracle',
        'description': 'Deutsch Jozsa Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dj_oracle_schema',
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
        nbits = math.log(len(bitmap),2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AlgorithmError('Input not the right length')
        self._nbits = int(nbits)

        # checks that the input bitstring represents a constant or balanced function
        function = False
        self._bitsum = sum([int(bit) for bit in bitmap.values()])

        if self._bitsum == 0 or self._bitsum == 2 ** self._nbits:
            self._function = "constant"
            function = True
        elif self._bitsum == 2 ** (self._nbits - 1):
            self._function = "balanced"
            function = True
        if function == False:
            raise AlgorithmError('Input is not a balanced or constant function')

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

        if self._bitsum == 0: # constant function of 0
            qc.iden(self._qr_ancilla)
        elif self._bitsum == 2 ** self._nbits: # constant function of 1
            qc.x(self._qr_ancilla)
        elif self._bitsum == 2 ** (self._nbits - 1): # balanced function
            # create a balanced oracle from the highest bitstring with value one
            parameter = 1
            for i in range(2 ** self._nbits - 1, 0, -1):
                bitstring = numpy.binary_repr(i, self._nbits)
                value = int(self._bitmap[bitstring])
                if value == 1:
                    parameter = i
                    break
            for i in range(self._nbits):
                if (parameter & (1 << i)):
                    qc.cx(self._qr_variable[i], self._qr_ancilla[0])       
        return qc
            
    def evaluate_classically(self, assignment):
        return self._function == assignment
        
    def interpret_measurement(self, measurement, *args, **kwargs):
        top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]
        top_measurement = int(top_measurement)
        if top_measurement == 0:
            return "constant"
        else:
            return "balanced"
