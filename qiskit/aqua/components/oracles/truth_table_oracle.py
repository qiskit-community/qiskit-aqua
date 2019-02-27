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
"""
The Truth Table-based Quantum Oracle.
"""

import logging
import operator
import math
import numpy as np
from functools import reduce

from pyeda.inter import exprvars, And, Xor
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import ESOP, get_prime_implicants, get_exact_covers
from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)


def is_power_of_2(num):
    return num != 0 and ((num & (num - 1)) == 0)


class TruthTableOracle(Oracle):

    CONFIGURATION = {
        'name': 'TruthTableOracle',
        'description': 'Truth Table Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'truth_table_oracle_schema',
            'type': 'object',
            'properties': {
                'bitmaps': {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        {
                            "type": "string"
                        }
                    ]
                },
                "optimization": {
                    "type": "string",
                    "default": "espresso",
                    'oneOf': [
                        {
                            'enum': [
                                'off',
                                'qm-dlx'
                            ]
                        }
                    ]
                },
                'mct_mode': {
                    'type': 'string',
                    'default': 'basic',
                    'oneOf': [
                        {
                            'enum': [
                                'basic',
                                'advanced',
                                'noancilla',
                            ]
                        }
                    ]
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, bitmaps, optimization='off', mct_mode='basic'):
        """
        Constructor for Truth Table-based Oracle

        Args:
            bitmaps (str or [str]): A single binary string or a list of binary strings representing the desired
                single- and multi-value truth table.
            optimization (str): Optimization mode to use for minimizing the circuit.
                Currently, besides no optimization ('off'), Aqua also supports a 'qm-dlx' mode,
                which uses the Quine-McCluskey algorithm to compute the prime implicants of the truth table,
                and then compute an exact cover to try to reduce the circuit.
            mct_mode (str): The mode to use when constructing multiple-control Toffoli.
        """

        self.validate(locals())
        super().__init__()

        self._mct_mode = mct_mode
        self._optimization = optimization

        if isinstance(bitmaps, str):
            bitmaps = [bitmaps]

        self._bitmaps = bitmaps

        # check that the input bitmaps length is a power of 2
        if not is_power_of_2(len(bitmaps[0])):
            raise AquaError('Length of any bitmap must be a power of 2.')
        for bitmap in bitmaps[1:]:
            if not len(bitmap) == len(bitmaps[0]):
                raise AquaError('Length of all bitmaps must be the same.')
        self._nbits = int(math.log(len(bitmaps[0]), 2))
        self._num_outputs = len(bitmaps)

        esop_exprs = []
        for bitmap in bitmaps:
            esop_expr = self._get_esop_ast(bitmap)
            esop_exprs.append(esop_expr)

        self._esops = [
            ESOP(esop_expr, num_vars=self._nbits) for esop_expr in esop_exprs
        ] if esop_exprs else None

        self.construct_circuit()

    def _get_esop_ast(self, bitmap):
        v = exprvars('v', self._nbits)

        def binstr_to_vars(binstr):
            return [
                       (~v[x[1] - 1] if x[0] == '0' else v[x[1] - 1])
                       for x in zip(binstr, reversed(range(1, self._nbits + 1)))
                   ][::-1]

        if self._optimization == 'off':
            expression = Xor(*[
                And(*binstr_to_vars(term)) for term in
                [np.binary_repr(idx, self._nbits) for idx, v in enumerate(bitmap) if v == '1']])
        else:  # self._optimization == 'qm-dlx':
            ones = [i for i, v in enumerate(bitmap) if v == '1']
            if not ones:
                return ('const', 0,)
            dcs = [i for i, v in enumerate(bitmap) if v == '*' or v == '-' or v.lower() == 'x']
            pis = get_prime_implicants(ones=ones, dcs=dcs)
            cover = get_exact_covers(ones, pis)[-1]
            clauses = []
            for c in cover:
                if len(c) == 1:
                    term = np.binary_repr(c[0], self._nbits)
                    clause = And(*[
                        v for i, v in enumerate(binstr_to_vars(term))
                    ])
                elif len(c) > 1:
                    c_or = reduce(operator.or_, c)
                    c_and = reduce(operator.and_, c)
                    _ = np.binary_repr(c_and ^ c_or, self._nbits)[::-1]
                    clause = And(*[
                        v for i, v in enumerate(binstr_to_vars(np.binary_repr(c_and, self._nbits))) if _[i] == '0'
                    ])
                else:
                    raise AquaError('Unexpected cover term size {}.'.format(len(c)))
                if clause:
                    clauses.append(clause)
            expression = Xor(*clauses)

        raw_ast = expression.to_ast()
        idx_mapping = {
            u: v + 1 for u, v in zip(sorted(expression.usupport), [v.indices[0] for v in sorted(expression.support)])
        }

        if raw_ast[0] == 'and' or raw_ast[0] == 'or' or raw_ast[0] == 'xor':
            clauses = []
            for c in raw_ast[1:]:
                if c[0] == 'lit':
                    clauses.append(('lit', (idx_mapping[c[1]]) if c[1] > 0 else (-idx_mapping[-c[1]])))
                elif (c[0] == 'or' or c[0] == 'and') and (raw_ast[0] != c[0]):
                    clause = []
                    for l in c[1:]:
                        clause.append(('lit', (idx_mapping[l[1]]) if l[1] > 0 else (-idx_mapping[-l[1]])))
                    clauses.append((c[0], *clause))
                else:
                    raise AquaError('Unrecognized logic expression: {}'.format(raw_ast))
        elif raw_ast[0] == 'const' or raw_ast[0] == 'lit':
            return raw_ast
        else:
            raise AquaError('Unrecognized root expression type: {}.'.format(raw_ast[0]))
        ast = (raw_ast[0], *clauses)
        return ast

    @property
    def variable_register(self):
        return self._variable_register

    @property
    def ancillary_register(self):
        return self._ancillary_register

    @property
    def output_register(self):
        return self._output_register

    def construct_circuit(self):
        if self._circuit is not None:
            return self._circuit
        self._circuit = QuantumCircuit()
        self._output_register = QuantumRegister(self._num_outputs, name='o')
        if self._esops:
            for i, e in enumerate(self._esops):
                if e is not None:
                    ci = e.construct_circuit(output_register=self._output_register, output_idx=i)
                    self._circuit += ci
            self._variable_register = self._ancillary_register = None
            for qreg in self._circuit.qregs:
                if qreg.name == 'v':
                    self._variable_register = qreg
                elif qreg.name == 'a':
                    self._ancillary_register = qreg
        else:
            self._variable_register = QuantumRegister(self._nbits, name='v')
            self._ancillary_register = None
            self._circuit.add_register(self._variable_register, self._output_register)
        return self._circuit

    def evaluate_classically(self, measurement):
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1], range(len(measurement)))]
        ret = [bitmap[int(measurement, 2)] == '1' for bitmap in self._bitmaps]
        if self._num_outputs == 1:
            return ret[0], assignment
        else:
            return ret, assignment
