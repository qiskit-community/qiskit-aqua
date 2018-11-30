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
from qiskit_aqua.algorithms.components.oracles import Oracle
from qiskit import QuantumRegister, QuantumCircuit 
import math
import numpy as np
import operator

logger = logging.getLogger(__name__)

class DeutschJozsaOracle(Oracle):

    DJO_CONFIGURATION = {
        'name': 'DeutschJozsaOracle',
        'description': 'Deutsch Jozsa Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dj_oracle_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DJO_CONFIGURATION.copy())
        self._qr_variable = None
        self._qr_ancilla = None
        self._qr_outcome = None
        
        self._function = None
        self._circuit = QuantumCircuit()

    def init_args(self, **args):
        pass
        
    def variable_register(self):
        return self._qr_variable

    def ancillary_register(self):
        return self._qr_ancilla

    def outcome_register(self):
        pass
    
    def circuit(self):
        return self._circuit
    
    def check_input(self, djo_input):
        # checks that the input bitstring length is a power of two
        nbits = math.log(len(djo_input),2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AlgorithmError('Input not the right length')
        nbits = int(nbits)

        # checks that the input bitstring represents a constant or balanced function
        function = False
        bitsum = sum([int(bit) for bit in djo_input.values()])
        if bitsum == 0 or bitsum == 2 ** nbits:
            self._function = "constant"
            function = True
        elif bitsum == 2 ** (nbits - 1):
            self._function = "balanced"
            function = True
        
        if function == False:
            raise AlgorithmError('Input is not a balanced or constant function')
    
    def construct_circuit(self, djo_input):
        nbits = int(math.log(len(djo_input),2))
        bitsum = sum([int(bit) for bit in djo_input.values()])
        
        self._qr_variable = QuantumRegister(nbits, name='v')
        self._qr_ancilla = QuantumRegister(1, name='a')

        self._circuit = QuantumCircuit(self._qr_variable, self._qr_ancilla)
        
        if bitsum == 0: # constant function of 0
            self._circuit.iden(self._qr_ancilla)
        elif bitsum == 2 ** nbits: # constant function of 1
            self._circuit.x(self._qr_ancilla)
        elif bitsum == 2 ** (nbits - 1): # balanced function
            # create a balanced oracle from the highest bitstring with value one
            parameter = 1
            for i in range(2 ** nbits - 1, 0, -1):
                bitstring = np.binary_repr(i, nbits)
                value = int(djo_input[bitstring])
                if value == 1:
                    parameter = i
                    break
            for i in range(nbits):
                if (parameter & (1 << i)):
                    self._circuit.cx(self._qr_variable[i], self._qr_ancilla[0])          
            
    def evaluate_classically(self, assignment):
        return self._function == assignment
        
    def interpret_measurement(self, measurement, *args, **kwargs):
        top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]
        top_measurement = int(top_measurement)
        if top_measurement == 0:
            return "constant"
        else:
            return "balanced"
