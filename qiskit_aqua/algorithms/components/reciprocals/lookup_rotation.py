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

"""Controlled rotation for the HHL algorithm based on partial table lookup"""

from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua.algorithms.components.reciprocals import Reciprocal
from qiskit_aqua.utils import cnx_na

import numpy as np
import itertools

import logging


logger = logging.getLogger(__name__)


class LookupRotation(Reciprocal):
    """Partial table lookup to rotate ancilla qubit"""

    PROP_EVO_TIME = 'evo_time'
    PROP_REG_SIZE = 'reg_size'
    PROP_PAT_LENGTH = 'pat_length'
    PROP_SUBPAT_LENGTH = 'subpat_length'
    PROP_NEGATIVE_EVALS = 'negative_evals'

    LOOKUP_CONFIGURATION = {
        'name': 'LOOKUP_ROT',
        'description': 'approximate inversion for HHL based on table lookup',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'reciprocal_lookup_schema',
            'type': 'object',
            'properties': {
                PROP_EVO_TIME: {
                    'type': 'number',
                },
                PROP_REG_SIZE: {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 2
                },
                PROP_PAT_LENGTH: {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 0
                },
                PROP_SUBPAT_LENGTH: {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 0
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'bool',
                    'default': False
                },
            },
            'additionalProperties': False
        },
    }

    def __init__(self, configuration=None):
        self._configuration = configuration or self.LOOKUP_CONFIGURATION.copy()
        self._anc = None
        self._workq = None
        self._msb = None
        self._ev = None
        self._circuit = None
        self._reg_size = 0
        self._pat_length = 0
        self._subpat_length = 0
        self._evo_time = 0
        self._negative_evals = False

    def init_args(self, evo_time, reg_size=0, pat_length=0, subpat_length=0, 
            negative_evals=False):
        self._evo_time = evo_time
        self._reg_size = reg_size
        self._pat_length = pat_length
        self._subpat_length = subpat_length
        self._negative_evals = negative_evals

    @staticmethod
    def classic_approx(k, n, m, negative_evals=False):
        '''Calculate error of arcsin rotation using k bits fixed
        point numbers and n bit accuracy'''

        def bin_to_num(binary):
            num = np.sum([2 ** -(n + 1)
                for n, i in enumerate(reversed(binary)) if i == '1'])
            return num

        def max_sign_bit(binary):
            return len(binary) - list(reversed(binary)).index('1')

        def min_lamb(k):
            return 2 ** -k

        def get_est_lamb(pattern, msb, n):
            '''Estimate the bin mid point and return the float value'''
            if msb - n > 0:
                pattern[msb - n - 1] = '1'
                return bin_to_num(pattern)
            else:
                return bin_to_num(pattern)

        from collections import OrderedDict
        output = OrderedDict()
        for msb in range(k - 1, n - 1, -1):
            #skip first bit if negative ev are used
            if negative_evals and msb==k-1 : continue
            vec = ['0'] * k
            vec[msb] = '1'
            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                msb_array = []

                for appendpat in itertools.product('10', repeat=n - m):
                    pattern = pattern_ + appendpat
                    vec[msb - n:msb] = pattern
                    e_l = get_est_lamb(vec.copy(), msb, n)
                    lambda_array.append(e_l)
                    msb_array.append(msb)
                    app_pattern_array.append(list(reversed(appendpat)))

                #rewrite MSB to correct index in QuantumRegister
                msb_num = k-msb-1
                if msb_num in list(output.keys()):
                    prev_res = output[msb_num]
                else:
                    prev_res = []
                
                output.update(
                    {msb_num: prev_res + [(list(reversed(pattern_)),
                                app_pattern_array, lambda_array)]})

        # last iterations
        vec = ['0'] * k
        for pattern_ in itertools.product('10', repeat=m):
            app_pattern_array = []
            lambda_array = []
            msb_array = []
            for appendpat in itertools.product('10', repeat=n - m):
                pattern = list(pattern_ + appendpat).copy()
                if '1' not in pattern and ( not negative_evals ):
                    continue
                elif '1' not in pattern and negative_evals:
                    e_l = 0.5
                else:
                    vec[msb - n:msb] = list(pattern)
                    e_l = get_est_lamb(vec.copy(), msb, n)
                lambda_array.append(e_l)
                msb_array.append(msb - 1)
                app_pattern_array.append(list(reversed(appendpat)))
                
            msb_num = k-msb
            if msb_num in list(output.keys()):
                prev_res = output[msb_num]
            else:
                prev_res = []
                
            output.update({msb_num: prev_res + [(list(reversed(pattern_)),
                    app_pattern_array, lambda_array)]})

        return output

    def _set_msb(self, msb, ev_reg, msb_num, last_iteration=False):
        qc = self._circuit
        ev = [ev_reg[i] for i in range(len(ev_reg))]
        #last_iteration = no MSB set, only the n-bit long pattern
        if last_iteration:
            if msb_num == 1:
                qc.x(ev[0])
                qc.cx(ev[0], msb[0])
                qc.x(ev[0])
            elif msb_num == 2:
                qc.x(ev[0])
                qc.x(ev[1])
                qc.ccx(ev[0], ev[1], msb[0])
                qc.x(ev[1])
                qc.x(ev[0])
            elif msb_num > 2:
                for idx in range(msb_num):
                    qc.x(ev[idx])
                qc.cnx(ev[:msb_num], msb[0])
                for idx in range(msb_num):
                    qc.x(ev[idx])
            else:
                qc.x(msb[0]) 
        elif msb_num == 0:
            qc.cx(ev[0], msb)
        elif msb_num == 1:
            qc.x(ev[0])
            qc.ccx(ev[0], ev[1], msb[0])
            qc.x(ev[0])
        elif msb_num > 1:
            for idx in range(msb_num):
                qc.x(ev[idx])
            qc.cnx(ev[:msb_num+1], msb[0])
            for idx in range(msb_num):
                qc.x(ev[idx])
        else:
            raise RuntimeError("MSB register index < 0")

    def _set_measurement(self):
        qc = self._circuit
        if 'c_ev' not in list(qc.regs.keys()):
            self.c_ev = ClassicalRegister(self._reg_size, 'c_ev')
            qc.add(self.c_ev)
        if 'c_anc' not in list(qc.regs.keys()):
            self.c_anc = ClassicalRegister(1, 'c_anc')
            qc.add(self.c_anc)
            
        qc.measure(self._anc, self.c_anc)
        qc.measure(self._ev, self.c_ev)

    def _set_bit_pattern(self, pattern, tgt, offset):
        qc = self._circuit
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])
        if len(pattern) > 2:
            temp_reg = [self._ev[i] for i in range(offset, offset+len(pattern))]
            qc.cnx(temp_reg, tgt)
        elif len(pattern) == 2:
            qc.ccx(self._ev[offset], self._ev[offset + 1], tgt)
        elif len(pattern) == 1:
            qc.cx(self._ev[offset], tgt)
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])

    def ccry(self, theta, control1, control2, target):
        '''Implement ccRy gate using no additional ancilla qubits'''
        # double angle because the rotation is defined with theta/2
        theta = 2 * theta
        qc = self._circuit
        theta_half = theta / 2
        qc.cu3(theta_half, 0, 0, control2, target)
        qc.cx(control1, control2)
        qc.cu3(- theta_half, 0, 0, control2, target)
        qc.cx(control1, control2)
        qc.cu3(theta_half, 0, 0, control1, target)

    def construct_circuit(self, mode, inreg):
        #initialize circuit
        if mode == "vector":
            raise NotImplementedError("mode vector not supported")
        self._ev = inreg
        self._workq = QuantumRegister(1, 'work')
        self._msb = QuantumRegister(1, 'msb')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._workq, self._msb, self._anc)
        self._circuit = qc
 
        m = self._subpat_length
        n = self._pat_length
        k = self._reg_size
        
        #get classically precomputed eigenvalue binning
        approx_dict = LookupRotation.classic_approx(k, n, m, negative_evals=self._negative_evals)
        
        old_msb = None
        # for negative EV, we pass a pseudo register ev[1:] ign. sign bit
        ev = [self._ev[i] for i in range(len(self._ev))]
        
        for _, msb in enumerate(list(approx_dict.keys())):
            #read m-bit and (n-m) bit patterns for current MSB and corr. Lambdas
            pattern_map = approx_dict[msb]
            #set most-significant-bit register and uncompute previous
            if self._negative_evals:
                if old_msb != msb:
                    if old_msb is not None:
                        self._set_msb(self._msb, ev[1:], int(old_msb-1))
                    old_msb = msb
                    if msb + n == k:
                        self._set_msb(self._msb, ev[1:], int(msb-1), last_iteration=True)
                    else:
                        self._set_msb(self._msb, ev[1:], int(msb-1), last_iteration=False)
            else:
                if old_msb != msb:
                    if old_msb is not None:
                        self._set_msb(self._msb, self._ev, int(old_msb))
                    old_msb = msb
                    if msb + n == k:
                        self._set_msb(self._msb, self._ev, int(msb), last_iteration=True)
                    else:
                        self._set_msb(self._msb, self._ev, int(msb), last_iteration=False)
            #offset = start idx for ncx gate setting and unsetting m-nit long bitstring
            offset_mpat = msb + (n - m) if msb < k - n else msb + n - m - 1
            for mainpat, subpat, lambda_ar in pattern_map:
                #set m-bit pattern in register workq
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)
                #iterate of all 2**(n-m) combinations for fixed m-bit
                for subpattern, lambda_ in zip(subpat, lambda_ar):
                    
                    #calculate rotation angle
                    theta =  np.arcsin(2 ** int(-k) / lambda_)
                    #offset for ncx gate checking subpattern
                    offset = msb + 1 if msb < k - n else msb
                
                    #rotation is happening here
                    #1. rotate by half angle
                    self.ccry(theta / 2, self._workq[0], self._msb[0], self._anc[0])
                    #2. cnx gate to reverse rotation direction
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                    #3. rotate by inverse of halfangle to uncompute / complete 
                    self.ccry(-theta / 2, self._workq[0], self._msb[0], self._anc[0])
                    #4. cnx gate to uncompute first cnx gate
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                #uncompute m-bit pattern
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)
                
        # uncompute msb register
        if self._negative_evals:
            self._set_msb(self._msb, ev[1:], int(msb-1), last_iteration=True)
        else:
            self._set_msb(self._msb, self._ev, int(msb), last_iteration=True)
            
        #rotate by pi to fix sign for negative evals
        if self._negative_evals:
            qc.cu3(2*np.pi,0,0,self._ev[0],self._anc[0])
        self._circuit = qc
        return self._circuit

    #
    # def _execute_rotation(self,shots):
    #     self._construct_rotation_circuit()
    #     shots = shots
    #     backend = self._backend
    #     if backend == "local_qasm_simulator" and shots == 1:
    #         self._circuit.snapshot("1")
    #         result = execute(self._circuit, backend=backend,
    #                          shots=shots,config={
    #                              "data":["hide_statevector","quantum_state_ket"]}).result()
    #         sv = result.get_data()["snapshots"]["1"]["quantum_state_ket"][0]
    #         res_dict = []
    #         for d in sv.keys():
    #                 if self._negative_evals:
    #                     num = sum([2 ** -(i + 2)
    #                         for i, e in enumerate(reversed(d.split()[-1][:-1])) if e == "1"])
    #                     if d.split()[-1][-1] == '1':
    #                         num *= -1
    #                         if '1' not in d.split()[-1][:-1]:
    #                             num = -0.5
    #                 else:
    #                     num = sum([2 ** -(i + 1)
    #                         for i, e in enumerate(reversed(d.split()[-1])) if e == "1"])
    #                 if self._evo_time is not None:
    #                     num *= 2*np.pi/self._evo_time
    #                 if d.split()[0] == '1':
    #                     res_dict.append(("Anc 1",num, sv[d][0],d))
    #                 else:
    #                     res_dict.append(("Anc 0",num, sv[d][0],d))
    #         self._ret = res_dict
    #         return res_dict
    #     elif backend == "local_qasm_simulator":
    #         self._set_measurement()
    #         result = execute(self._circuit,
    #                          backend=backend, shots=shots).result()
    #
    #         rd = result.get_counts(self._circuit)
    #         rets = sorted([[rd[k], k, k] for k in rd])[::-1]
    #         # return rets
    #         for d in rets:
    #             print(d)
    #             d[0] /= shots
    #             d[0] = np.sqrt(d[0])
    #             # split registers which are white space separated and decode
    #             c1, c2 = d[2].split()
    #             c2_ = sum([2 ** -(i + 1)
    #                        for i, e in enumerate(reversed(c2)) if e == "1"])
    #             d[2] = ' '.join([c1, str(c2_)])
    #         self._ret = rets
    #         return rets
    #     elif backend == "local_statevector_simulator":
    #         result = execute(self._circuit, "local_statevector_simulator")
    #         sv = result.result().get_data()["statevector"]
    #         self._ret = sv
    #         return sv
    #     else:
    #         raise RuntimeError("Backend not implemented yet") 
