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
from qiskit_aqua.components.reciprocals import Reciprocal
import numpy as np


class LongDivision(Reciprocal):
    "Finds reciprocal with long division method and rotates the ancilla qubit"

    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_SCALE = 'scale'
    PROP_PRECISION = 'precision'
    PROP_EVO_TIME = 'evo_time'
    PROP_LAMBDA_MIN = 'lambda_min'

    CONFIGURATION = {
        'name': 'LongDivision',
        'description': 'reciprocal computation with long division and rotation of the ancilla qubit',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'reciprocal_long_division_schema',
            'type': 'object',
            'properties': {
                PROP_NUM_ANCILLAE: {
                    'type': ['integer', 'null'],
                    'default': None,
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },
                PROP_SCALE:{
                    'type': 'number',
                    'default': 1,
                },
                PROP_PRECISION:{
                    'type': ['integer', 'null'],
                    'default': None,                    
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

    def __init__(self, num_ancillae=None, scale=1, precision=None,
                 evo_time=None, lambda_min=None, negative_evals=False):
        self.validate(locals())
        super().__init__()
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._scale = scale
        self._precision = precision
        self._evo_time = evo_time
        self._lambda_min = lambda_min
        self._circuit = None
        self._ev = None
        self._rec = None
        self._anc = None
        self._reg_size = 0
        self._neg_offset = 0
        self._n = 0
        
    @classmethod
    def init_params(cls, params):
        num_ancillae = params.get(LongDivision.PROP_NUM_ANCILLAE)
        negative_evals = params.get(LongDivision.PROP_NEGATIVE_EVALS)
        scale = params.get(LongDivision.PROP_SCALE)
        precision = params.get(LongDivision.PROP_PRECISION)
        evo_time = params.get(LongDivision.PROP_EVO_TIME)
        lambda_min = params.get(LongDivision.PROP_LAMBDA_MIN)

        return cls(num_ancillae=num_ancillae, negative_evals=negative_evals,
                   scale=scale, precision=precision, evo_time=evo_time,
                   lambda_min=lambda_min)

    def _ld_circuit(self):
        
        def subtract(a, b, b0, c, z,r, rj, n):
            qc = QuantumCircuit(a, b0, b, c, z, r)
            qc2 = QuantumCircuit(a, b0, b ,c, z,r)                
               
            def subtract_in(qc, a, b, b0, c , z, r, n):
                """subtraction realized with ripple carry adder"""
                
                def maj(p, a, b, c):
                    p.cx(c, b)
                    p.cx(c, a)
                    p.ccx(a, b, c)
            
                def uma(p, a, b, c):
                    p.ccx(a, b, c)
                    p.cx(c, a)
                    p.cx(a, b)
                    
                for i in range(n):
                    qc.x(a[i])                
                maj(qc, c[0], a[0], b[n-2])    
                 
                for i in range(n-2):
                    maj(qc, b[n-2-i+self._neg_offset], a[i+1], b[n-3-i+self._neg_offset])  
                    
                maj(qc, b[self._neg_offset+0], a[n-1], b0[0])                                 
                qc.cx(a[n-1], z[0])                             
                uma(qc, b[self._neg_offset+0], a[n-1], b0[0])                    
                
                for i in range(2, n):
                    uma(qc, b[self._neg_offset+i-1], a[n-i], b[self._neg_offset+i-2])                  
                
                uma(qc, c[0], a[0], b[n-2+self._neg_offset])          
                
                for i in range(n):
                    qc.x(a[i])                
                
                qc.x(z[0])             
                        
            def u_maj(p, a, b, c, r):
                p.ccx(c, r, b)
                p.ccx(c, r, a)
                p.cnx_na([r, a, b], c)
                
            def u_uma(p, a, b, c, r):
                p.cnx_na([r, a, b], c)
                p.ccx(c,r, a)
                p.ccx(a, r, b)
            
            def unsubtract(qc2, a, b, b0, c, z, r, n):
                """controlled inverse subtraction to uncompute the registers(when
                the result of the subtraction is negative)"""
                
                for i in range(n):
                    qc2.cx(r, a[i])     
                u_maj(qc2, c[0], a[0], b[n-2],r)                      
                
                for i in range(n-2):
                    u_maj(qc2, b[n-2-i+self._neg_offset], a[i+1], b[n-3-i+self._neg_offset], r)            
                
                u_maj(qc2, b[self._neg_offset+0], a[n-1], b0[0], r)        
                qc2.ccx(a[n-1],r, z[0])            
                u_uma(qc2, b[self._neg_offset+0], a[n-1], b0[0], r)
                
                for i in range(2, n):
                    u_uma(qc2, b[self._neg_offset+i-1], a[n-i], b[self._neg_offset+i-2], r)            
                
                u_uma(qc2, c[0], a[0], b[n-2+self._neg_offset], r)      
                
                for i in range(n):
                    qc2.cx(r, a[i]) 
                
                un_qc = qc2.reverse()        
                un_qc.cx(r, z[0])   
                return un_qc
            
            #ASSEMBLING CIRCUIT FOR CONTROLLED SUBTRACTION:
            subtract_in(qc, a, b, b0, c, z, r[rj], n)
            qc.x(a[n-1])
            qc.cx(a[n-1], r[rj])
            qc.x(a[n-1])
            
            qc.x(r[rj])
            qc += unsubtract(qc2, a, b, b0, c, z, r[rj], n)
            qc.x(r[rj])
            
            return qc       
    
        def shift_to_one(qc, b, anc, n):
            '''controlled bit shifting for the initial alignment of the most 
            significant bits'''
            
            for i in range(n-2):            #set all the anc1 qubits to 1
                qc.x(anc[i])
            
            for j2 in range(n-2):           #if msb is 1, change ancilla j2 to 0
                qc.cx(b[0+self._neg_offset], anc[j2])                            
                for i in  np.arange(0,n-2):
                    i = int(i)                  #which activates shifting with the 2 Toffoli gates
                    qc.ccx(anc[j2], b[i+1+self._neg_offset], b[i+self._neg_offset])
                    qc.ccx(anc[j2], b[i+self._neg_offset], b[i+1+self._neg_offset]) 
                                       
            for i in range(n-2):                #negate all the anc1
                qc.x(anc[i])
                              
        def shift_one_left(qc, b, n):   
            for i in np.arange(n-1,0, -1):
                i = int(i)
                qc.cx(b[i-1], b[i]) 
                qc.cx(b[i], b[i-1])            

        def shift_one_leftc(qc, b, ctrl, n):
            for i in np.arange(n-2,0, -1):
                i = int(i)
                qc.ccx(ctrl, b[i-1], b[i])
                qc.ccx(ctrl, b[i], b[i-1])                            
            return qc
        
        def shift_one_rightc(qc, b, ctrl, n):                   
            for i in np.arange(0, n-1):
                i = int(i)
                qc.ccx(ctrl, b[n-2-i+self._neg_offset], b[n-1-i+self._neg_offset])
                qc.ccx(ctrl, b[n-1-i+self._neg_offset], b[n-2-i+self._neg_offset])
                  
        # executing long division:
        self._circuit.x(self._a[self._n-2])
        shift_to_one(self._circuit, self._ev, self._anc1, self._n)  #initial alignment of most significant bits

        for rj in range(self._precision): #iterated subtraction and shifting
            self._circuit += subtract(self._a, self._ev, self._b0, self._c,
                                      self._z, self._rec, rj, self._n)
            shift_one_left(self._circuit, self._a, self._n)
               
        for ish in range(self._n-2): #unshifting due to initial alignment
            shift_one_leftc(self._circuit, self._rec, self._anc1[ish],
                            self._precision + self._num_ancillae)
            self._circuit.x(self._anc1[ish])
            shift_one_rightc(self._circuit, self._ev, self._anc1[ish], self._num_ancillae)
            self._circuit.x(self._anc1[ish])

    def _rotation(self):
        qc = self._circuit
        rec_reg = self._rec
        ancilla = self._anc

        if self._negative_evals:
            for i in range(0, self._precision + self._num_ancillae):
                qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[i], ancilla)
            qc.cu3(2*np.pi, 0, 0,self._ev[0], ancilla)  #correcting the sign
        else:
            for i in range(0, self._precision + self._num_ancillae):
                qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[i], ancilla)
                
        self._circuit = qc
        self._rec = rec_reg
        self._anc = ancilla
        
    def construct_circuit(self, mode, inreg):
        
        if mode == "vector":
            raise NotImplementedError("mode vector not supported")
        self._ev = inreg

        if self._scale == 0:
            self._scale = 2**-len(inreg)

        if self._negative_evals:
            self._neg_offset = 1

        self._num_ancillae = len(self._ev) - self._neg_offset
        self._n = self._num_ancillae + 1  
        
        if self._num_ancillae < 3:
            raise NotImplementedError("Min. size of eigenregister is 3 for positive eigenvalues and 4 when negative eigenvalues are enabled")
        
        if self._precision is None:
            self._precision = self._num_ancillae 
                  
        self._a = QuantumRegister(self._n, 'one') #register storing 1
        self._b0 = QuantumRegister(1, 'b0') #extension of b - required by subtraction
        self._anc1 = QuantumRegister(self._num_ancillae-1, 'algn_anc') # ancilla for the initial shifting
        self._z = QuantumRegister(1, 'z') #subtraction overflow
        self._c = QuantumRegister(1, 'c') #carry
        self._rec = QuantumRegister(self._precision + self._num_ancillae, 'res') #reciprocal result
        self._anc = QuantumRegister(1, 'anc')                        
        qc = QuantumCircuit(self._a, self._b0, self._ev, self._anc1, self._c, 
                            self._z, self._rec, self._anc)
       
        self._circuit = qc
        self._ld_circuit()
        self._rotation()
             
        return self._circuit
