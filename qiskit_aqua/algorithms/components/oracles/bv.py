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

class BernsteinVaziraniOracle(Oracle):

    BVO_CONFIGURATION = {
        'name': 'BernsteinVaziraniOracle',
        'description': 'Bernstein Vazirani Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'bv_oracle_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.BVO_CONFIGURATION.copy())
        self._qr_variable = None
        self._qr_ancilla = None
        self._qr_outcome = None
        
        self._parameter = ""
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
    
    def check_input(self, bvo_input):
        # checks that the input bitstring length is a power of two
        nbits = math.log(len(bvo_input),2)
        if math.ceil(nbits) != math.floor(nbits):
            raise AlgorithmError('Input not the right length')
        nbits = int(nbits)

        # figure out the hidden parameter
        for i in range(nbits-1,-1,-1):
            bitstring = np.binary_repr(2**i, nbits)
            bit = bvo_input[bitstring]
            self._parameter += bit
            
    def construct_circuit(self, bvo_input):
        nbits = int(math.log(len(bvo_input),2))
        
        self._qr_variable = QuantumRegister(nbits, name='v')
        self._qr_ancilla = QuantumRegister(1, name='a')

        self._circuit = QuantumCircuit(self._qr_variable, self._qr_ancilla)
        for i in range(nbits):
            if (int(self._parameter) & (1 << i)):
                self._circuit.cx(self._qr_variable[i], self._qr_ancilla[0])          
            
    def evaluate_classically(self, assignment):
        return self._parameter == assignment
        
    def interpret_measurement(self, measurement, *args, **kwargs):
        return max(measurement.items(), key=operator.itemgetter(1))[0]