# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Controlled rotation for the HHL algorithm based on partial table lookup"""

import itertools
import logging
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.circuits.gates import mct  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class LookupRotation(Reciprocal):

    """The Lookup Rotation for Reciprocals.

    A calculation of reciprocals of eigenvalues is performed and controlled
    rotation of ancillary qubit via a lookup method. It uses a partial table
    lookup of rotation angles to rotate an ancillary qubit by arcsin(C/lambda).
    Please refer to the HHL documentation for an explanation of this method.
    """

    CONFIGURATION = {
        'name': 'Lookup',
        'description': 'approximate inversion for HHL based on table lookup',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'reciprocal_lookup_schema',
            'type': 'object',
            'properties': {
                'pat_length': {
                    'type': ['integer', 'null'],
                    'default': None,
                },
                'subpat_length': {
                    'type': ['integer', 'null'],
                    'default': None,
                },
                'negative_evals': {
                    'type': 'boolean',
                    'default': False
                },
                'scale': {
                    'type': 'number',
                    'default': 0,
                    'minimum': 0,
                    'maximum': 1,
                },
                'evo_time': {
                    'type': ['number', 'null'],
                    'default': None
                },
                'lambda_min': {
                    'type': ['number', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
    }

    def __init__(
            self,
            pat_length=None,
            subpat_length=None,
            scale=0,
            negative_evals=False,
            evo_time=None,
            lambda_min=None
    ):
        """Constructor.

        Args:
            pat_length (int, optional): the number of qubits used for binning pattern
            subpat_length (int, optional): the number of qubits used for binning sub-pattern
            scale (float, optional): the scale of rotation angle, corresponds to HHL constant C
            negative_evals (bool, optional): indicate if negative eigenvalues need to be handled
            evo_time (float, optional): the evolution time
            lambda_min (float, optional): the smallest expected eigenvalue
        """

        self.validate(locals())
        super().__init__()
        self._pat_length = pat_length
        self._subpat_length = subpat_length
        self._negative_evals = negative_evals
        self._scale = scale
        self._evo_time = evo_time
        self._lambda_min = lambda_min
        self._anc = None
        self._workq = None
        self._msq = None
        self._ev = None
        self._circuit = None
        self._reg_size = 0

    def sv_to_resvec(self, statevector, num_q):
        half = int(len(statevector) / 2)
        vec = statevector[half:half + 2 ** num_q]
        return vec

    @staticmethod
    def _classic_approx(k, n, m, negative_evals=False):
        """Approximate arcsin(1/x) for controlled-rotation.

        This method calculates the binning of arcsin(1/x) function using k
        bits fixed point numbers and n bit accuracy.

        Args:
            k (int): register length
            n (int): num bits following most-significant qubit taken into account
            m (int): length of sub string of n-qubit pattern
            negative_evals (bool): flag for using first qubit as sign bit

        Returns:
            dict: Dictionary containing values of approximated and binned values.
        """

        def bin_to_num(binary):
            """Convert to numeric"""
            num = np.sum([2 ** -(n + 1) for n, i in enumerate(reversed(
                binary)) if i == '1'])
            return num

        def get_est_lamb(pattern, fo, n, k):
            """Estimate the bin mid point and return the float value"""
            if fo - n > 0:
                remainder = sum([2 ** -i for i in range(k - (fo - n - 1),
                                                        k + 1)])
                return bin_to_num(pattern)+remainder/2
            return bin_to_num(pattern)
        # pylint: disable=import-outside-toplevel
        from collections import OrderedDict
        output = OrderedDict()
        fo = None
        for fo in range(k - 1, n - 1, -1):
            # skip first bit if negative ev are used
            if negative_evals and fo == k - 1:
                continue
            # init bit string
            vec = ['0'] * k
            # set most significant bit
            vec[fo] = '1'
            # iterate over all 2^m combinations = sub string in n-bit pattern
            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                fo_array = []
                # iterate over all 2^(n-m) combinations
                for appendpat in itertools.product('10', repeat=n - m):
                    # combine both generated patterns
                    pattern = pattern_ + appendpat
                    vec[fo - n:fo] = pattern
                    # estimate bin mid point
                    e_l = get_est_lamb(vec.copy(), fo, n, k)
                    lambda_array.append(e_l)
                    fo_array.append(fo)
                    app_pattern_array.append(list(reversed(appendpat)))

                # rewrite first-one to correct index in QuantumRegister
                fo_pos = k-fo-1
                if fo_pos in list(output.keys()):
                    prev_res = output[fo_pos]
                else:
                    prev_res = []

                output.update(
                    {fo_pos: prev_res + [(list(reversed(pattern_)),
                                          app_pattern_array, lambda_array)]})

        # last iterations, only last n bits != 0
        last_fo = fo
        vec = ['0'] * k
        for pattern_ in itertools.product('10', repeat=m):
            app_pattern_array = []
            lambda_array = []
            fo_array = []
            for appendpat in itertools.product('10', repeat=n - m):
                pattern = list(pattern_ + appendpat).copy()
                if '1' not in pattern and (not negative_evals):
                    continue
                if '1' not in pattern and negative_evals:
                    e_l = 0.5
                else:
                    vec[last_fo - n:last_fo] = list(pattern)
                    e_l = get_est_lamb(vec.copy(), last_fo, n, k)
                lambda_array.append(e_l)
                fo_array.append(last_fo - 1)
                app_pattern_array.append(list(reversed(appendpat)))

            fo_pos = k-last_fo
            if fo_pos in list(output.keys()):
                prev_res = output[fo_pos]
            else:
                prev_res = []

            output.update({fo_pos: prev_res + [(list(reversed(pattern_)),
                                                app_pattern_array,
                                                lambda_array)]})

        return output

    def _set_msq(self, msq, ev_reg, fo_pos, last_iteration=False):
        """Adds multi-controlled NOT gate to entangle |msq> qubit
        with states having the correct first-one qubit

        Args:
            msq (QuantumRegister): most-significant qubit, this is a garbage qubit
            ev_reg (QuantumRegister): register storing eigenvalues
            fo_pos (int): position of first-one bit
            last_iteration (bool): switch which is set for numbers where only the
                        last n bits is different from 0 in the binary string
        Raises:
            RuntimeError: invalid input
        """
        qc = self._circuit
        ev = [ev_reg[i] for i in range(len(ev_reg))]
        # last_iteration = no MSQ set, only the n-bit long pattern
        if last_iteration:
            if fo_pos == 1:
                qc.x(ev[0])
                qc.cx(ev[0], msq[0])
                qc.x(ev[0])
            elif fo_pos == 2:
                qc.x(ev[0])
                qc.x(ev[1])
                qc.ccx(ev[0], ev[1], msq[0])
                qc.x(ev[1])
                qc.x(ev[0])
            elif fo_pos > 2:
                for idx in range(fo_pos):
                    qc.x(ev[idx])
                qc.mct(ev[:fo_pos], msq[0], None, mode='noancilla')
                for idx in range(fo_pos):
                    qc.x(ev[idx])
            else:
                qc.x(msq[0])
        elif fo_pos == 0:
            qc.cx(ev[0], msq)
        elif fo_pos == 1:
            qc.x(ev[0])
            qc.ccx(ev[0], ev[1], msq[0])
            qc.x(ev[0])
        elif fo_pos > 1:
            for idx in range(fo_pos):
                qc.x(ev[idx])
            qc.mct(ev[:fo_pos + 1], msq[0], None, mode='noancilla')
            for idx in range(fo_pos):
                qc.x(ev[idx])
        else:
            raise RuntimeError("first-one register index < 0")

    def _set_bit_pattern(self, pattern, tgt, offset):
        """Add multi-controlled NOT gate to circuit that has negated/normal
        controls according to the pattern specified

        Args:
            pattern (list): List of strings giving a bit string that negates
                controls if '0'
            tgt (QuantumRegister): target qubit
            offset (int): start index for the control qubits
        """
        qc = self._circuit
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])
        if len(pattern) > 2:
            temp_reg = [self._ev[i]
                        for i in range(offset, offset+len(pattern))]
            qc.mct(temp_reg, tgt, None, mode='noancilla')
        elif len(pattern) == 2:
            qc.ccx(self._ev[offset], self._ev[offset + 1], tgt)
        elif len(pattern) == 1:
            qc.cx(self._ev[offset], tgt)
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])

    def construct_circuit(self, mode, inreg):  # pylint: disable=arguments-differ

        """Construct the Lookup Rotation circuit.

        Args:
            mode (str): construction mode, 'matrix' not supported
            inreg (QuantumRegister): input register, typically output register of Eigenvalues

        Returns:
            QuantumCircuit: containing the Lookup Rotation circuit.
         Raises:
            NotImplementedError: mode not supported
        """

        # initialize circuit
        if mode == 'matrix':
            raise NotImplementedError('The matrix mode is not supported.')
        if self._lambda_min:
            self._scale = self._lambda_min/2/np.pi*self._evo_time
        if self._scale == 0:
            self._scale = 2**-len(inreg)
        self._ev = inreg
        self._workq = QuantumRegister(1, 'work')
        self._msq = QuantumRegister(1, 'msq')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._ev, self._workq, self._msq, self._anc)
        self._circuit = qc
        self._reg_size = len(inreg)
        if self._pat_length is None:
            if self._reg_size <= 6:
                self._pat_length = self._reg_size - \
                                   (2 if self._negative_evals else 1)
            else:
                self._pat_length = 5
        if self._reg_size <= self._pat_length:
            self._pat_length = self._reg_size - \
                               (2 if self._negative_evals else 1)
        if self._subpat_length is None:
            self._subpat_length = int(np.ceil(self._pat_length/2))
        m = self._subpat_length
        n = self._pat_length
        k = self._reg_size
        neg_evals = self._negative_evals

        # get classically precomputed eigenvalue binning
        approx_dict = LookupRotation._classic_approx(k, n, m,
                                                     negative_evals=neg_evals)

        fo = None
        old_fo = None
        # for negative EV, we pass a pseudo register ev[1:] ign. sign bit
        ev = [self._ev[i] for i in range(len(self._ev))]

        for _, fo in enumerate(list(approx_dict.keys())):
            # read m-bit and (n-m) bit patterns for current first-one and
            # correct Lambdas
            pattern_map = approx_dict[fo]
            # set most-significant-qbit register and uncompute previous
            if self._negative_evals:
                if old_fo != fo:
                    if old_fo is not None:
                        self._set_msq(self._msq, ev[1:], int(old_fo - 1))
                    old_fo = fo
                    if fo + n == k:
                        self._set_msq(self._msq, ev[1:], int(fo - 1),
                                      last_iteration=True)
                    else:
                        self._set_msq(self._msq, ev[1:], int(fo - 1),
                                      last_iteration=False)
            else:
                if old_fo != fo:
                    if old_fo is not None:
                        self._set_msq(self._msq, self._ev, int(old_fo))
                    old_fo = fo
                    if fo + n == k:
                        self._set_msq(self._msq, self._ev, int(fo),
                                      last_iteration=True)
                    else:
                        self._set_msq(self._msq, self._ev, int(fo),
                                      last_iteration=False)
            # offset = start idx for ncx gate setting and unsetting m-bit
            # long bitstring
            offset_mpat = fo + (n - m) if fo < k - n else fo + n - m - 1
            for mainpat, subpat, lambda_ar in pattern_map:
                # set m-bit pattern in register workq
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)
                # iterate of all 2**(n-m) combinations for fixed m-bit
                for subpattern, lambda_ in zip(subpat, lambda_ar):

                    # calculate rotation angle
                    theta = 2*np.arcsin(min(1, self._scale / lambda_))
                    # offset for ncx gate checking subpattern
                    offset = fo + 1 if fo < k - n else fo

                    # rotation is happening here
                    # 1. rotate by half angle
                    qc.mcry(theta / 2, [self._workq[0], self._msq[0]],
                            self._anc[0], None, mode='noancilla')
                    # 2. mct gate to reverse rotation direction
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                    # 3. rotate by inverse of halfangle to uncompute / complete
                    qc.mcry(-theta / 2, [self._workq[0], self._msq[0]],
                            self._anc[0], None, mode='noancilla')
                    # 4. mct gate to uncompute first mct gate
                    self._set_bit_pattern(subpattern, self._anc[0], offset)
                # uncompute m-bit pattern
                self._set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)

        last_fo = fo
        # uncompute msq register
        if self._negative_evals:
            self._set_msq(self._msq, ev[1:], int(last_fo - 1),
                          last_iteration=True)
        else:
            self._set_msq(self._msq, self._ev, int(last_fo),
                          last_iteration=True)

        # rotate by pi to fix sign for negative evals
        if self._negative_evals:
            qc.cu3(2*np.pi, 0, 0, self._ev[0], self._anc[0])
        self._circuit = qc
        return self._circuit
