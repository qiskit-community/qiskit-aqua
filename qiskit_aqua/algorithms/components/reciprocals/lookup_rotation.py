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

import numpy as np
import itertools

import logging


logger = logging.getLogger(__name__)


class LookupRotation(Reciprocal):
    """Partial table lookup to rotate ancilla qubit"""

    PROP_PAT_LENGTH = 'pat_length'
    PROP_SUBPAT_LENGTH = 'subpat_length'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_SCALE = 'scale'
    PROP_EVO_TIME = 'evo_time'
    PROP_LAMBDA_MIN = 'lambda_min'

    LOOKUP_CONFIGURATION = {
        'name': 'LOOKUP',
        'description': 'approximate inversion for HHL based on table lookup',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'reciprocal_lookup_schema',
            'type': 'object',
            'properties': {
                PROP_PAT_LENGTH: {
                    'type': ['integer', 'null'],
                    'default': None,
                },
                PROP_SUBPAT_LENGTH: {
                    'type': ['integer', 'null'],
                    'default': None,
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },
                PROP_SCALE: {
                    'type': 'number',
                    'default': 0,
                    'minimum': 0,
                    'maximum': 1,
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
        self._configuration = configuration or self.LOOKUP_CONFIGURATION.copy()
        self._anc = None
        self._workq = None
        self._msb = None
        self._ev = None
        self._circuit = None
        self._reg_size = 0
        self._pat_length = None
        self._subpat_length = None
        self._negative_evals = False
        self._scale = 0
        self._evo_time = None
        self._lambda_min = None

    def init_args(self, pat_length=None, subpat_length=None, scale=0,
            negative_evals=False, evo_time=None, lambda_min=None):
        self._pat_length = pat_length
        self._subpat_length = subpat_length
        self._negative_evals = negative_evals
        self._scale = scale
        self._evo_time = evo_time
        self._lambda_min = lambda_min

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
                qc.cnx_na(ev[:msb_num], msb[0])
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
            qc.cnx_na(ev[:msb_num+1], msb[0])
            for idx in range(msb_num):
                qc.x(ev[idx])
        else:
            raise RuntimeError("MSB register index < 0")

    def _set_bit_pattern(self, pattern, tgt, offset):
        qc = self._circuit
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])
        if len(pattern) > 2:
            temp_reg = [self._ev[i] for i in range(offset, offset+len(pattern))]
            qc.cnx_na(temp_reg, tgt)
        elif len(pattern) == 2:
            qc.ccx(self._ev[offset], self._ev[offset + 1], tgt)
        elif len(pattern) == 1:
            qc.cx(self._ev[offset], tgt)
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])

    def construct_circuit(self, mode, inreg):
        #initialize circuit
        if mode == "vector":
            raise NotImplementedError("mode vector not supported")
        if self._lambda_min:
            self._scale = self._lambda_min/2/np.pi*self._evo_time
        if self._scale == 0:
            self._scale = 2**-len(inreg)
        self._ev = inreg
        self._workq = QuantumRegister(1, 'work')
        self._msb = QuantumRegister(1, 'msb')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._workq, self._msb, self._anc)
        self._circuit = qc
        self._reg_size = len(inreg)
        if self._pat_length is None:
            if self._reg_size <= 6:
                self._pat_length = self._reg_size - (2 if self._negative_evals else 1)
            else:
                self._pat_length = 5
        if self._reg_size <= self._pat_length:
            self._pat_length = self._reg_size - (2 if self._negative_evals else 1)
        if self._subpat_length is None:
            self._subpat_length = int(np.ceil(self._pat_length/2))
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
                    theta =  2*np.arcsin(min(1, self._scale / lambda_))
                    #offset for ncx gate checking subpattern
                    offset = msb + 1 if msb < k - n else msb
                
                    #rotation is happening here
                    #1. rotate by half angle
                    qc.cnu3(theta/2, 0, 0, [self._workq[0], self._msb[0]], self._anc[0])
                    #2. cnx_na gate to reverse rotation direction
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                    #3. rotate by inverse of halfangle to uncompute / complete 
                    qc.cnu3(-theta/2, 0, 0, [self._workq[0], self._msb[0]], self._anc[0])
                    #4. cnx_na gate to uncompute first cnx_na gate
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
