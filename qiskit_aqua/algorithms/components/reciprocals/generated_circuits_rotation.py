from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua.algorithms.components.reciprocals import Reciprocal

import numpy as np
import math

import logging

logger = logging.getLogger(__name__)

class GeneratedCircuit(Reciprocal):
    "Pregenerated Circuits to compute reciprocal and rotate ancilla qubit"

    PROP_NUM_ANCILLAE = 'num_ancillae'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_SCALE = 'scale'

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
                    'default': None,
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },
                PROP_SCALE:{
                    'type': 'number'
                    'default': 1,    
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
        self._offset = 0

    def init_args(self, num_ancillae=0, scale=0,
            negative_evals=False):
        self._num_ancillae = num_ancillae
        self._negative_evals = negative_evals
        self._scale = scale


    @staticmethod
    def _nc_toffoli(qc, ev_reg, rec_reg, ctl,tgt,ctlnumber, n):
    '''Implement n+1-bit toffoli using the approach in Elementary gates'''
        
        assert ctlnumber>=3 #"This method works only for more than 2 control bits"
        w = n-1
        from sympy.combinatorics.graycode import GrayCode
        gray_code = list(GrayCode(ctlnumber).generate_gray())
        last_pattern = None
            #angle to construct nth square root of diagonlized pauli x matrix
            #via u1(lam_angle)
        lam_angle = np.pi/(2**(ctlnumber-1))
            #transform to eigenvector basis of pauli X
        qc.h(tgt)
        for pattern in gray_code:
            
            if not '1' in pattern:
                continue
            if last_pattern is None:
                last_pattern = pattern
                #find left most set bit
            lm_pos = list(pattern).index('1')
            #find changed bit
            comp = [i!=j for i,j in zip(pattern,last_pattern)]
            if True in comp:
                pos = comp.index(True)
            else:
                pos = None
            if pos is not None:
                if pos != lm_pos:
                    if ctl[pos] >= n and ctl[lm_pos] >= n:
                        qc.cx(rec_reg[ctl[pos]-n], rec_reg[ctl[lm_pos]-n])
                    elif ctl[pos]<n and ctl[lm_pos] <n:
                        qc.cx(ev_reg[w-ctl[pos]], ev_reg[w-ctl[lm_pos]])
                    elif ctl[pos]<n and ctl[lm_pos] >= n:
                        qc.cx(ev_reg[w-ctl[pos]], rec_reg[ctl[lm_pos]-n])
                    else:
                        qc.cx(rec_reg[ctl[pos]-n], ev_reg[w-ctl[lm_pos]])
                else:
                    indices = [i for i, x in enumerate(pattern) if x == '1']
                    for idx in indices[1:]:
                        if ctl[idx] >= n and ctl[lm_pos] >= n:
                            qc.cx(rec_reg[ctl[idx]-n], rec_reg[ctl[lm_pos]-n])
                        elif ctl[idx]<n and ctl[lm_pos] <n:
                            qc.cx(ev_reg[w-ctl[idx]], ev_reg[w-ctl[lm_pos]])
                        elif ctl[idx]<n and ctl[lm_pos] >= n:
                            qc.cx(ev_reg[w-ctl[idx]], rec_reg[ctl[lm_pos]-n])
                        else:
                            qc.cx(rec_reg[ctl[idx]-n], ev_reg[w-ctl[lm_pos]])
                #check parity
            if pattern.count('1') % 2 == 0:
                #inverse
                if ctl[lm_pos] < n:
                    qc.cu1(-lam_angle,ev_reg[w-ctl[lm_pos]],tgt)
                else:
                    qc.cu1(-lam_angle,rec_reg[ctl[lm_pos]-n],tgt)
            else:
                if ctl[lm_pos] < n:
                    qc.cu1(lam_angle,ev_reg[w-ctl[lm_pos]],tgt)
                else:
                    qc.cu1(lam_angle,rec_reg[ctl[lm_pos]-n],tgt)
            last_pattern = pattern
        qc.h(tgt)

    def _parse_circuit(self):
        # Parse the pre-generated circuit with specified number of qubits

        n = self._num_ancillae
        qc = self._circuit
        ev_reg = self._ev[self._offset:]
        rec_reg = self._rec

        with open("intdiv-esop0-rec{}.txt" .format(n), "r") as f:
            data = f.readlines()

        w = n-1
        for i in range(len(data)):
            ctl = []
            xgate = []
            doubledigit = False
            sign = False
            if data[i][1].isdigit():
                ctlnumber = int(data[i][0] + data[i][1])-1
            else:
                ctlnumber = int(data[i][0])-1
            if int(data[i][-2]) >= n:
                tgt = rec_reg[int(data[i][-2]) - n]
            else:
                tgt = ev_reg[w-int(data[i][-2])]
            if data[i][-3].isdigit():
                if int(data[i][-3] + data[i][-2]) >= n:
                    tgt = rec_reg[int(data[i][-3] + data[i][-2]) - n]
                else:
                    tgt = ev_reg[w-int(data[i][-3] + data[i][-2])]           
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
            ctl.pop(0)
            ctl.pop(-1)
            if ctlnumber == 0: #single not gate
                qc.x(tgt)
            elif ctlnumber == 1: #cnot gate
                if ctl[0] >= n:
                    qc.cx(rec_reg[ctl[0] - n], tgt)
                else:
                    qc.cx(ev_reg[w-ctl[0]], tgt)
            elif ctlnumber == 2: #toffoli gate
                if ctl[0] >= n and ctl[1] >= n:
                    qc.ccx(rec_reg[ctl[0]-n], rec_reg[ctl[1]-n], tgt)
                elif ctl[0]<n and ctl[1] <n:
                    qc.ccx(ev_reg[w-ctl[0]], ev_reg[w-ctl[1]], tgt)
                elif ctl[0]<n and ctl[1] >= n:
                    qc.ccx(ev_reg[w-ctl[0]], rec_reg[ctl[1]-n], tgt)
                else:
                    qc.ccx(rec_reg[ctl[0]-n], ev_reg[w-ctl[1]], tgt)
            else: #not gates controlled with more than 2 qubits
                nc_toffoli(qc, ev_reg, rec_reg, ctl, tgt, ctlnumber, n)
            for j in xgate:
                if j >= n:
                    qc.x(rec_reg[j-n])
                else:
                    qc.x(ev_reg[w-j])

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
            qc.cu3(2*np.pi,0,0,self._ev[0], ancilla)
        else:
            for i in range(1, n+1):
	            qc.cu3(self._scale*2**(-i), 0, 0, rec_reg[n-i], ancilla)

        self._circuit = qc
        self._rec = reg_rec
        self._anc = ancilla

        
    def construct_circuit(self, inreg):
        #initialize circuit
        self._ev = inreg
        self._rec = QuantumRegister(self._num_ancillae, 'reciprocal')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._rec, self._anc)
        self._circuit = qc
        if self._negative_evals:
            self._offset = 1
        self._parse_circuit()
        self._rotation()

        return self._circuit