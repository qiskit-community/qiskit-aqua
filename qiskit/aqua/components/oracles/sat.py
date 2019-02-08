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
The SAT (Satisfiability) Quantum Oracle.
"""

import logging

from qiskit.aqua.components.oracles import Oracle
from qiskit.aqua.utils import CNF

logger = logging.getLogger(__name__)


class SAT(Oracle):

    CONFIGURATION = {
        'name': 'SAT',
        'description': 'Satisfiability Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'sat_oracle_schema',
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

    def __init__(self, dimacs_cnf, mct_mode='basic'):
        self.validate(locals())
        super().__init__()
        self._mct_mode = mct_mode

        ls = [
            l.strip() for l in dimacs_cnf.split('\n')
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
        self._cnf = CNF(cnf_expr)

        if not h_nv == self._cnf.num_variables:
            logger.warning('Inaccurate variable count {} in cnf header, actual count is {}.'.format(h_nv, nv))
        if not h_nc == self._cnf.num_clauses:
            logger.warning('Inaccurate clause count {} in cnf header, actual count is {}.'.format(h_nc, nc))

    @property
    def variable_register(self):
        return self._cnf.qr_variable

    @property
    def ancillary_register(self):
        return self._cnf.qr_ancilla

    @property
    def outcome_register(self):
        return self._cnf.qr_outcome

    def construct_circuit(self):
        return self._cnf.construct_circuit(mct_mode=self._mct_mode)

    def evaluate_classically(self, assignment):
        assignment_set = set(assignment)
        for clause in self._cnf.expr:
            if assignment_set.isdisjoint(clause):
                return False
        return True

    def interpret_measurement(self, top_measurement=None):
        return [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(top_measurement[::-1], range(len(top_measurement)))]
