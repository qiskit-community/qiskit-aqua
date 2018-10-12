#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:27:04 2018

@author: gawel
"""

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

import sys
sys.path.append('../')
from qiskit_aqua.algorithms.components.reciprocals import Reciprocal
from qiskit_aqua.utils import cnx_na, cnu3
from qiskit_aqua.utils.cnx_no_anc import CNXGate


import numpy as np
import math
import os

#import logging

#logger = logging.getLogger(__name__)

class LongDivision(Reciprocal):
    "Finds recirpocal with long division method and rotates the ancilla qubit"

    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_SCALE = 'scale'
    PROP_PRECISION = 'precision'

    LONGDIVISION_CONFIGURATION = {
        'name': 'LONGDIVISION',
        'description': 'reciprocal computation with long division and rotation of the ancilla qubit',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'generated_circuits_reciprocal_schema',
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
                    'default':1,                    
                },
                PROP_PRECISION:{
                    'type': 'number',
                    'default':4,                    
                }
            },
            'additionalProperties': False
        },
        }

    def __init__(self, configuration=None):
        self._configuration = configuration or self.LONGDIVISION_CONFIGURATION.copy()
        self._num_ancillae = None
        self._circuit = None
        self._ev = None
        self._rec = None
        self._anc = None
        self._reg_size = 0
        self._negative_evals = False
        self._scale = 0
        self._neg_offset = 0

        self._precision = 0
        self._offset =0
        self._n = 0
        
        
    def init_args(self, num_ancillae=0, scale=0,precision =0,
            negative_evals=False):
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._scale = scale
        self._precision = precision
        
        
    def _ld_circuit(self):
        #self._offset = self._n - 2
        k_r = self._precision
        a = self._a 
        b = self._ev
        b0 = self._b0
        anc_s = self._anc_s
        anc = self._anc1
        z = self._z
        c = self._c
        r = self._rec
        
        qc = self._circuit 
               
        def subtract(a, b,b0, c, z,r, rj):                         
            qc = QuantumCircuit(a , b, b0, c, z, r)
            qc2 = QuantumCircuit(a, b, b0, c, z,r)                
               
            def subtract_in(qc, a, b, b0, c ,z, r):
                def maj(p, a, b, c):
                    p.cx(c, b)
                    p.cx(c, a)
                    p.ccx(a, b, c)
            
                def uma(p, a, b, c):
                    p.ccx(a, b, c)
                    p.cx(c, a)
                    p.cx(a, b)
                    
                for i in range(self._n):
                    qc.x(a[i])                
                maj(qc, c[0], a[0], b[0])                     
                for i in range(self._n-2):
                    maj(qc, b[i], a[i+1], b[i+1])                
                maj(qc, b[self._n-2], a[self._n-1], b0[0])                 
                qc.cx(a[self._n-1], z[0])             
                uma(qc, b[self._n-2], a[self._n-1], b0[0])                    
                for i in range(2, self._n-1):
                    uma(qc, b[self._n-i-1], a[self._n-i], b[self._n-i])                  
                uma(qc, c[0], a[0], b[0])          
                for i in range(self._n):
                    qc.x(a[i])                
                qc.x(z[0])             
                        
            def u_maj(p, a, b, c,r):
                p.ccx(c, r, b)
                p.ccx(c, r, a)
                p.cnx([r, a, b], c)
                
            def u_uma(p, a, b, c, r):
                p.cnx([r, a, b], c)
                p.ccx(c,r, a)
                p.ccx(a, r, b)
            
            def unsubtract(qc2, a, b, b0, c ,z, r):       
                for i in range(self._n):
                    qc2.cx(r, a[i])     
                u_maj(qc2, c[0], a[0], b[0],r)                      
                
                for i in range(self._n-2):
                    u_maj(qc2, b[i], a[i+1], b[i+1], r)            
                
                u_maj(qc2, b[self._n-2], a[self._n-1], b0[0], r)        
                qc2.ccx(a[self._n-1],r, z[0])            
                u_uma(qc2, b[self._n-2], a[self._n-1], b0[0], r)
                
                for i in range(2, self._n):
                    u_uma(qc2, b[self._n-i-1], a[self._n-i], b[self._n-i], r)            
                
                u_uma(qc2, c[0], a[0], b[0], r)      
                
                for i in range(self._n):
                    qc2.cx(r, a[i]) 
                
                un_qc = qc2.reverse()        
                un_qc.cx(r, z[0])   
                return un_qc
            
            subtract_in(qc, a, b,b0,  c ,z, r[rj])
            qc.x(a[self._n-1])
            qc.cx(a[self._n-1], r[rj])
            qc.x(a[self._n-1])
            
            qc.x(r[rj])
            qc += unsubtract(qc2, a, b,b0,  c ,z, r[rj])
            qc.x(r[rj])
            return qc       
    
        def shift_to_one(qc,b, anc, n):           
            for i in range(n-2):
                qc.x(anc[i])
            
            for j2 in range(n-2):
                qc.cx(b[n-2], anc[j2])                            
                for i in  np.arange(n-2,0, -1):
                    i = int(i)
                    qc.ccx(anc[j2], b[i-1], b[i])
                    qc.ccx(anc[j2], b[i], b[i-1])      
                    
            for i in range(n-2):
                qc.x(anc[i])
            
                    
        def shift_one_left(qc, b, n):   
            for i in  np.arange(n-1,0, -1):
                i = int(i)
                qc.cx(b[i-1], b[i]) 
                qc.cx(b[i], b[i-1])            
        
                
        def shift_one_leftc(qc, b, b0, ctrl, n):            
            qc.ccx(ctrl, b[n-1], b0)
            qc.ccx(ctrl, b0, b[n-1])        
            for i in  np.arange(n-1,0, -1):
                i = int(i)
                qc.ccx(ctrl, b[i-1], b[i])
                qc.ccx(ctrl, b[i], b[i-1])  
            
            qc.ccx(ctrl, b[n-1], b0)
            return qc    

    
        qc.x(a[self._n-2])               
        shift_to_one(qc,b, anc, self._n) 
        
        for rj in range(self._precision):                
            qc += subtract(a, b, b0, c, z,r, rj)                       
            shift_one_left(qc, a, self._n)
        
        for ish in range(self._n-2):
            shift_one_leftc(qc, r, anc_s[ish], anc[ish] , k_r)   

        self._circuit = qc    
        #return qc
        
        
        
    def _rotation(self):
        #Make rotation on ancilla qubit
        qc = self._circuit
        rec_reg = self._rec
        ancilla = self._anc

        if self._negative_evals:
            for i in range(1,self._n+1):
	            qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[self._n-i], ancilla)
            qc.cu3(2*np.pi,0,0,self._ev[0], ancilla) #correcting the sign
        else:
            for i in range(1, self._n+1):
	            qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[self._n-i], ancilla)

        self._circuit = qc
        self._rec = rec_reg
        self._anc = ancilla

        
    def construct_circuit(self, mode, inreg, precision):
        
        #initialize circuit
        if mode == "vector":
            raise NotImplementedError("mode vector not supported")
        self._ev = inreg
        if self._negative_evals:
            self._neg_offset = 1        
        self._num_ancillae = len(self._ev) - self._neg_offset


        self._n = self._num_ancillae + 1
        self._offset = self._n - 2
        self._precision = precision
        
        self._a = QuantumRegister(self._n, 'one')                       #register storing 1
        self._b0 = QuantumRegister(1, 'b0')                             #extension of b - required by subtraction
        self._anc_s = QuantumRegister(self._offset, 'shifting_ancilla') #ancilla for the result shifting
        self._anc1 = QuantumRegister(self._offset, 'aligning_ancilla')  #ancilla for the initial shifting
        self._z = QuantumRegister(1, 'z')                               #subtraction overflow
        self._c = QuantumRegister(1, 'c')                               #carry
        self._rec = QuantumRegister(self._precision, 'res')             #reciprocal result
        self._anc = QuantumRegister(1, 'anc')                           #HHL ancilla bit
        
        
        qc = QuantumCircuit(self._a, self._b0, self._ev, self._anc1, self._anc_s, self._c, 
                            self._z,self._rec, self._anc)
        
        self._circuit = qc
        self._ld_circuit()
        
        self._rotation()

        return self._circuit