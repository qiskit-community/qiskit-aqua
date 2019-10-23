# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The General Logical Expression-based Quantum Oracle.
"""

import logging
import re

from sympy.parsing.sympy_parser import parse_expr
from sympy.logic.boolalg import to_cnf, BooleanTrue, BooleanFalse
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import AquaError
from qiskit.aqua.circuits import CNF, DNF
from .oracle import Oracle
from .ast_utils import get_ast

logger = logging.getLogger(__name__)


class LogicalExpressionOracle(Oracle):
    """ LOgical expression Oracle """
    CONFIGURATION = {
        'name': 'LogicalExpressionOracle',
        'description': 'Logical Expression Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'logical_expression_oracle_schema',
            'type': 'object',
            'properties': {
                'expression': {
                    'type': ['string', 'null'],
                    'default': None
                },
                "optimization": {
                    "type": "boolean",
                    "default": False,
                },
                'mct_mode': {
                    'type': 'string',
                    'default': 'basic',
                    'enum': [
                        'basic',
                        'basic-dirty-ancilla',
                        'advanced',
                        'noancilla'
                    ]
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, expression=None, optimization=False, mct_mode='basic'):
        """
        Constructor.

        Args:
            expression (str): The string of the desired logical expression.
                It could be either in the DIMACS CNF format,
                or a general boolean logical expression, such as 'a ^ b' and 'v[0] & (~v[1] | v[2])'
            optimization (bool): Boolean flag for attempting logical expression optimization
            mct_mode (str): The mode to use for building Multiple-Control Toffoli.
        Raises:
            AquaError: invalid input
        """
        self.validate(locals())
        super().__init__()

        if expression is None:
            raise AquaError('Missing logical expression.')

        self._mct_mode = mct_mode.strip().lower()
        self._optimization = optimization

        expression = re.sub('(?i)' + re.escape(' and '), ' & ', expression)
        expression = re.sub('(?i)' + re.escape(' xor '), ' ^ ', expression)
        expression = re.sub('(?i)' + re.escape(' or '), ' | ', expression)
        expression = re.sub('(?i)' + re.escape('not '), '~', expression)

        orig_expression = expression
        # try parsing as normal logical expression that sympy recognizes
        try:
            raw_expr = parse_expr(expression)
        except Exception:  # pylint: disable=broad-except
            # try parsing as dimacs cnf
            try:
                expression = LogicalExpressionOracle._dimacs_cnf_to_expression(expression)
                raw_expr = parse_expr(expression)
            except Exception:
                raise AquaError('Failed to parse the input expression: {}.'.format(orig_expression))
        self._expr = raw_expr
        self._process_expr()
        self.construct_circuit()

    @staticmethod
    def _dimacs_cnf_to_expression(dimacs):
        lines = [
            ll for ll in [
                l.strip().lower() for l in dimacs.strip().split('\n')
            ] if len(ll) > 0 and not ll[0] == 'c'
        ]

        if not lines[0][:6] == 'p cnf ':
            raise AquaError('Unrecognized dimacs cnf header {}.'.format(lines[0]))

        def create_var(cnf_tok):
            return ('~v' + cnf_tok[1:]) if cnf_tok[0] == '-' else ('v' + cnf_tok)

        clauses = []
        for line in lines[1:]:
            toks = line.split()
            if not toks[-1] == '0':
                raise AquaError('Unrecognized dimacs line {}.'.format(line))

            clauses.append('({})'.format(' | '.join(
                [create_var(t) for t in toks[:-1]]
            )))
        return ' & '.join(clauses)

    def _process_expr(self):
        self._num_vars = len(self._expr.binary_symbols)
        self._lit_to_var = [None] + sorted(self._expr.binary_symbols, key=str)
        self._var_to_lit = dict(zip(self._lit_to_var[1:], range(1, self._num_vars + 1)))
        cnf = to_cnf(self._expr, simplify=self._optimization)

        if isinstance(cnf, BooleanTrue):
            ast = 'const', 1
        elif isinstance(cnf, BooleanFalse):
            ast = 'const', 0
        else:
            ast = get_ast(self._var_to_lit, cnf)

        if ast[0] == 'or':
            self._nf = DNF(ast, num_vars=self._num_vars)
        else:
            self._nf = CNF(ast, num_vars=self._num_vars)

    @property
    def variable_register(self):
        """ returns variable register """
        return self._variable_register

    @property
    def ancillary_register(self):
        """ returns ancillary register """
        return self._ancillary_register

    @property
    def output_register(self):
        """ returns output register """
        return self._output_register

    def construct_circuit(self):
        """ construct circuit """
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
        """ evaluate classically """
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1],
                                                                       range(len(measurement)))]
        assignment_dict = dict()
        for v in assignment:
            assignment_dict[self._lit_to_var[abs(v)]] = bool(v > 0)
        return self._expr.subs(assignment_dict), assignment
