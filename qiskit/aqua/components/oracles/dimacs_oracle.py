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
The DIMACS-based Quantum Oracle.
"""

import logging

from qiskit.aqua.components.oracles import Oracle
from qiskit.aqua.utils import CNF

logger = logging.getLogger(__name__)


class DimacsOracle(Oracle):

    CONFIGURATION = {
        'name': 'DimacsOracle',
        'description': 'Dimacs Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dimacs_oracle_schema',
            'type': 'object',
            'properties': {
                'dimacs_cnf': {
                    'type': 'string',
                },
                'mct_mode': {
                    'type': 'string',
                    'default': 'basic',
                    'oneOf': [
                        {'enum': [
                            'basic',
                            'advanced',
                            'noancilla'
                        ]}
                    ]
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, dimacs_str=None, mct_mode='basic'):
        """
        Constructor.

        Args:
            dimacs_str (str): The string representation in DIMACS format
            mct_mode (str): The mode to use for building Multiple-Control Toffoli
        """

        self.validate(locals())
        self._mct_mode = mct_mode

        if dimacs_str:
            ls = [
                l.strip() for l in dimacs_str.split('\n')
                if len(l) > 0 and not l.strip()[0] == 'c'
            ]
            headers = [l for l in ls if l[0] == 'p']
            if len(headers) == 1:
                p, sig, nv, nc = headers[0].split()
                assert p == 'p' and sig == 'cnf'
            else:
                raise ValueError('Invalid cnf format for SAT.')
            h_nv, h_nc = int(nv), int(nc)
            cs = [
                c.strip()
                for c in ' '.join(
                    [l for l in ls if not l[0] == 'p']
                ).split(' 0') if len(c) > 0
            ]
            cnf_expr = [
                [int(v) for v in c.split() if not int(v) == 0]
                for c in cs
                if (
                    len(c.replace('0', '')) > 0
                ) and (
                    '0' <= c[0] <= '9' or c[0] == '-'
                )
            ]
        else:
            raise ValueError('Missing input DIMACS string.')

        self._cnf = CNF(cnf_expr)
        super().__init__()

    @property
    def variable_register(self):
        return self._cnf.variable_register

    @property
    def ancillary_register(self):
        return self._cnf.ancillary_register

    @property
    def output_register(self):
        return self._cnf.output_register

    def construct_circuit(self):
        return self._cnf.construct_circuit(mct_mode=self._mct_mode)

    def evaluate_classically(self, measurement):
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1], range(len(measurement)))]
        assignment_set = set(assignment)
        for clause in self._cnf.expr:
            if assignment_set.isdisjoint(clause):
                return False, assignment
        return True, assignment
