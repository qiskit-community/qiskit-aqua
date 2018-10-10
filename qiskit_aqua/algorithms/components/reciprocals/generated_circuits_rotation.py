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

from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua.algorithms.components.reciprocals import Reciprocal
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt
import numpy as np
import math
import os

import logging

logger = logging.getLogger(__name__)

class GeneratedCircuit(Reciprocal):
    "Pregenerated Circuits to compute reciprocal and rotate ancilla qubit"

    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_SCALE = 'scale'
    PROP_EVO_TIME = 'evo_time'
    PROP_LAMBDA_MIN = 'lambda_min'

    GENCIRCUITS_CONFIGURATION = {
        'name': 'GENCIRCUITS',
        'description': 'reciprocal computation and rotation with pregenerated circuits',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'generated_circuits_reciprocal_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_ANCILLAE: {
                    'type': ['integer', 'null'],
                    'default': None
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },
                PROP_SCALE: {
                    'type': 'number',
                    'default':0,
                    'minimum':0,
                    'maximum':1,
                },
                PROP_EVO_TIME: {
                    'type': ['number', 'null'],
                    'default': None
                },
                PROP_LAMBDA_MIN: {
                    'type': ['number', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
}

    def __init__(self, configuration=None):
        self._configuration = configuration or self.GENCIRCUITS_CONFIGURATION.copy()
        self._num_ancillae = None
        self._circuit = None
        self._ev = None
        self._rec = None
        self._anc = None
        self._reg_size = 0
        self._negative_evals = False
        self._scale = 0
        self._lambda_min = None
        self._evo_time = None
        self._offset = 0
        self._rec_circuit = None
        self._inverse = None

    def init_args(self, num_ancillae=0, scale=0, evo_time = None, lambda_min = None,
            negative_evals=False):
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._scale = scale
        self._evo_time = evo_time
        self._lambda_min = lambda_min

    def _parse_circuit(self):
        # Parse the pre-generated circuit with specified number of qubits

        n = self._num_ancillae
        qc = self._circuit
        ev_reg = self._ev
        rec_reg = self._rec
        offset = self._offset
        #Getting the files of the pregenerated circuits
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path+"/GenCirc/intdiv-esop0-rec{}.txt" .format(n), "r") as f:
            data = f.readlines()

        #for negative eigenvalues, we ignore the first bit in the eigenvalue register
        w = n-1+offset
        #parsing the file; first number is number of qubits needed for gate; last number is controlled qubit; inbetween controlling qubits
        for i in range(len(data)):
            ctl = []
            xgate = []
            doubledigit = False
            sign = False
            #setting number of control qubits, needed for gate type
            if data[i][1].isdigit():
                ctlnumber = int(data[i][0] + data[i][1])-1
            else:
                ctlnumber = int(data[i][0])-1
            #setting target qubit, checking if number is > 10
            if int(data[i][-2]) >= n:
                tgt = rec_reg[int(data[i][-2]) - n]
            else:
                tgt = ev_reg[w-int(data[i][-2])]
            if data[i][-3].isdigit():
                if int(data[i][-3] + data[i][-2]) >= n:
                    tgt = rec_reg[int(data[i][-3] + data[i][-2]) - n]
                else:
                    tgt = ev_reg[w-int(data[i][-3] + data[i][-2])]          
            #saving control qubits, checking if they're double digits
            for j in range(len(data[i])):
                if data[i][j] == str(" "):
                    doubledigit = False
                elif data[i][j] == str("-"):
                    sign = True
                elif data[i][j].isdigit() and not data[i][j+1].isdigit() and not doubledigit:
                    ctl.append(int(data[i][j]))
                    if sign:
                        xgate.append(int(data[i][j]))
                        if int(data[i][j]) >= n:
                            qc.x(rec_reg[int(data[i][j]) - n])
                        else:
                            qc.x(ev_reg[w-int(data[i][j])])
                        sign = False
                elif data[i][j].isdigit() and data[i][j+1].isdigit():
                    ctl.append(int(data[i][j] + data[i][j+1]))
                    doubledigit = True
                    if sign:
                        xgate.append(int(data[i][j] + data[i][j+1]))
                        if int(data[i][j]+ data[i][j+1]) >= n:
                            qc.x(rec_reg[int(data[i][j] + data[i][j+1]) - n])
                        else:
                            qc.x(ev_reg[w-int(data[i][j] + data[i][j+1])])               
                        sign = False
            #deleting entry for control number and target qubit
            ctl.pop(0)
            ctl.pop(-1)
            #checking which qubit belongs in which register
            for i in range(len(ctl)):
                if ctl[i] < n:
                    ctl[i] = ev_reg[w-ctl[i]]
                else:
                    ctl[i] = rec_reg[ctl[i] - n]
            if ctlnumber == 0: #single not gate
                qc.x(tgt)
            elif ctlnumber == 1: #cnot gate
                qc.cx(ctl[0], tgt)
            elif ctlnumber == 2: #toffoli gate
                qc.ccx(ctl[0], ctl[1], tgt)
            else: #not gates controlled with more than 2 qubits
                qc.cnx_na(ctl, tgt)
            for j in xgate:
                if j >= n:
                    qc.x(rec_reg[j-n])
                else:
                    qc.x(ev_reg[w-j])
        
        self._rec_circuit = qc
        self._circuit = qc
        self._rec = rec_reg
        
    def _rotation(self):
        #Make rotation on ancilla qubit
        qc = self._circuit
        n = self._num_ancillae
        rec_reg = self._rec
        ancilla = self._anc

        if self._negative_evals:
            for i in range(1, n+1):
	            qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[n-i], ancilla)
            qc.cu3(2*np.pi,0,0,self._ev[0], ancilla) #correcting the sign
        else:
            for i in range(1, n+1):
	            qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[n-i], ancilla)

        self._circuit = qc
        self._rec = rec_reg
        self._anc = ancilla

        
    def _reverse_parsing(self):
    #reversing the reciprocal to make the circuit reversible
        qc = QuantumCircuit(self._ev, self._rec, self._anc)
        #deleting entries for rotation
        if self._negative_evals:
            cutoff = len(self._rec_circuit.data) - (self._num_ancillae + 1)
        else: 
            cutoff = len(self._rec_circuit.data) - self._num_ancillae
        for gate in reversed(self._rec_circuit.data[:cutoff]):
            gate.reapply(qc)
        return qc


    def construct_circuit(self, mode, inreg):
        #initialize circuit
        if mode == "vector":
            raise NotImplementedError("mode vector not supported")
        #setting the scaling for the rotation - scale has to be proportional to smallest eigenvalue, if that is not known, smallest possible value is set
        if self._lambda_min is not None:
            self._scale = self._lambda_min/2/np.pi*self._evo_time
        if self._scale == 0:
            self._scale = 2**(-len(inreg))
        self._ev = inreg
        #checking for negative eigenvalues and deleting first qubit out of eigenvalue register, if true
        if self._negative_evals:
            self._offset = 1        
        self._num_ancillae = len(self._ev) - self._offset
        if self._num_ancillae < 5:
            raise NotImplementedError("eigenvalue register has to contain at least 5 bits")
        self._rec = QuantumRegister(self._num_ancillae, 'reciprocal')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._rec, self._anc)
        self._circuit = qc
        
        self._parse_circuit()
        self._rotation()
        self._circuit += self._reverse_parsing()

        return self._circuit
