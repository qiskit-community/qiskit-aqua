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

from pyeda.inter import espresso_exprs
from pyeda.boolalg.bfarray import exprvars
from pyeda.boolalg.expr import ast2expr, AndOp, OrOp
from pyeda.parsing.dimacs import parse_cnf
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua import AquaError
from qiskit.aqua.utils import CNF, DNF
from qiskit.aqua.components.oracles import Oracle

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
                "optimization_mode": {
                    "anyOf": [
                        {
                            "type": "null",
                        },
                        {
                            "type": "string",
                            "default": "espresso",
                            'oneOf': [
                                {
                                    'enum': [
                                        'espresso',
                                    ]
                                }
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
                                'noancilla'
                            ]
                        }
                    ]
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, dimacs_str=None, optimization_mode=None, mct_mode='basic'):
        """
        Constructor.

        Args:
            dimacs_str (str): The string representation in DIMACS format.
            optimization_mode (string): Optimization mode to use for minimizing the circuit.
                Currently, besides no optimization if omitted, Aqua also supports an 'expresso' mode
                <https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer>
            mct_mode (str): The mode to use for building Multiple-Control Toffoli.
        """

        self.validate(locals())
        self._mct_mode = mct_mode

        if dimacs_str is None:
            raise ValueError('Missing input DIMACS string.')
        expr = ast2expr(parse_cnf(dimacs_str.strip()))
        self._num_vars = expr.degree

        if optimization_mode is None:
            self._nf = CNF(expr, num_vars=self._num_vars, depth=2)
        elif optimization_mode == 'espresso':
            expr_dnf = expr.to_dnf()
            if expr_dnf.is_zero() or expr_dnf.is_one():
                self._nf = CNF(expr_dnf, num_vars=self._num_vars)
            else:
                expr_dnf_m = espresso_exprs(expr_dnf)[0]
                if isinstance(expr_dnf_m, AndOp):
                    self._nf = CNF(expr_dnf_m, num_vars=self._num_vars)
                elif isinstance(expr_dnf_m, OrOp):
                    self._nf = DNF(expr_dnf_m, num_vars=self._num_vars)
                else:
                    raise AquaError('Unexpected espresso optimization result expr: {}'.format(expr_dnf_m))

        super().__init__()

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
        if self._circuit is None:
            if self._nf is not None:
                self._circuit = self._nf.construct_circuit(mct_mode=self._mct_mode)
                self._variable_register = self._nf.variable_register
                self._output_register = self._nf.output_register
                self._ancillary_register = self._nf.ancillary_register
            else:
                self._variable_register = QuantumRegister(self._num_vars, name='v')
                self._output_register = QuantumRegister(1, name='o')
                self._ancillary_register = None
                self._circuit = QuantumCircuit(self._variable_register, self._output_register)
        return self._circuit

    def evaluate_classically(self, measurement):
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1], range(len(measurement)))]
        if self._nf._expr.is_zero():
            found = False
        elif self._nf._expr.is_one():
            found = True
        else:
            prime_implicants = self._nf._expr.complete_sum()
            if isinstance(prime_implicants, AndOp):
                sols = [[l[1] for l in prime_implicants.to_ast()[1:]]]
            elif isinstance(prime_implicants, OrOp):
                sols =[[l[1] for l in c[1:]] for c in self._nf._expr.complete_sum().to_ast()[1:]]
            else:
                raise AquaError('Unexpected solution: {}'.format(prime_implicants))
            found = assignment in sols
        return found, assignment
