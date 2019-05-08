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
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua import AquaError
from qiskit.aqua.circuits import CNF, DNF
from .oracle import Oracle
from ._pyeda_check import _check_pluggable_valid as check_pyeda_valid
logger = logging.getLogger(__name__)


class LogicalExpressionOracle(Oracle):

    CONFIGURATION = {
        'name': 'LogicalExpressionOracle',
        'description': 'Logical Expression Oracle',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'logical_expression_oracle_schema',
            'type': 'object',
            'properties': {
                'expression': {
                    'type': ['string', 'null'],
                    'default': None
                },
                "optimization": {
                    "type": "string",
                    "default": "off",
                    'oneOf': [
                        {
                            'enum': [
                                'off',
                                'espresso'
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
                                'basic-dirty-ancilla',
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

    def __init__(self, expression=None, optimization='off', mct_mode='basic'):
        """
        Constructor.

        Args:
            expression (str): The string of the desired logical expression.
                It could be either in the DIMACS CNF format,
                or a general boolean logical expression, such as 'a ^ b' and 'v[0] & (~v[1] | v[2])'
            optimization (str): The mode of optimization to use for minimizing the circuit.
                Currently, besides no optimization ('off'), Aqua also supports an 'espresso' mode
                <https://en.wikipedia.org/wiki/Espresso_heuristic_logic_minimizer>
            mct_mode (str): The mode to use for building Multiple-Control Toffoli.
        """
        self.validate(locals())
        super().__init__()

        self._mct_mode = mct_mode
        self._optimization = optimization

        from pyeda.boolalg.expr import ast2expr, expr
        from pyeda.parsing.dimacs import parse_cnf
        if expression is None:
            raw_expr = expr(None)
        else:
            try:
                raw_expr = expr(expression)
            except:
                try:
                    raw_expr = ast2expr(parse_cnf(expression.strip(), varname='v'))
                except:
                    raise AquaError('Failed to parse the input expression: {}.'.format(expression))

        self._expr = raw_expr
        self._process_expr()
        self.construct_circuit()

    @staticmethod
    def check_pluggable_valid():
        check_pyeda_valid(LogicalExpressionOracle.CONFIGURATION['name'])

    @staticmethod
    def _normalize_literal_indices(raw_ast, raw_indices):
        idx_mapping = {r: i + 1 for r, i in zip(sorted(raw_indices), range(len(raw_indices)))}
        if raw_ast[0] == 'and' or raw_ast[0] == 'or':
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
                    raise AquaError('Unrecognized logical expression: {}'.format(raw_ast))
        elif raw_ast[0] == 'const' or raw_ast[0] == 'lit':
            return raw_ast
        else:
            raise AquaError('Unrecognized root expression type: {}.'.format(raw_ast[0]))
        return (raw_ast[0], *clauses)

    def _process_expr(self):
        from pyeda.inter import espresso_exprs
        from pyeda.boolalg.expr import AndOp, OrOp, Variable
        self._num_vars = self._expr.degree
        ast = self._expr.to_ast() if self._expr.is_cnf() else self._expr.to_cnf().to_ast()
        ast = LogicalExpressionOracle._normalize_literal_indices(ast, self._expr.usupport)

        if self._optimization == 'off':
            if ast[0] == 'or':
                self._nf = DNF(ast, num_vars=self._num_vars)
            else:
                self._nf = CNF(ast, num_vars=self._num_vars)
        else:  # self._optimization == 'espresso':
            expr_dnf = self._expr.to_dnf()
            if expr_dnf.is_zero() or expr_dnf.is_one():
                self._nf = CNF(('const', 0 if expr_dnf.is_zero() else 1), num_vars=self._num_vars)
            else:
                expr_dnf_m = espresso_exprs(expr_dnf)[0]
                expr_dnf_m_ast = LogicalExpressionOracle._normalize_literal_indices(
                    expr_dnf_m.to_ast(), expr_dnf_m.usupport
                )
                if isinstance(expr_dnf_m, AndOp) or isinstance(expr_dnf_m, Variable):
                    self._nf = CNF(expr_dnf_m_ast, num_vars=self._num_vars)
                elif isinstance(expr_dnf_m, OrOp):
                    self._nf = DNF(expr_dnf_m_ast, num_vars=self._num_vars)
                else:
                    raise AquaError('Unexpected espresso optimization result expr: {}'.format(expr_dnf_m))

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
        from pyeda.boolalg.expr import AndOp, OrOp, Variable
        assignment = [(var + 1) * (int(tf) * 2 - 1) for tf, var in zip(measurement[::-1], range(len(measurement)))]
        if self._expr.is_zero():
            return False, assignment
        elif self._expr.is_one():
            return True, assignment
        else:
            prime_implicants = self._expr.complete_sum()
            if prime_implicants.is_zero():
                sols = []
            elif isinstance(prime_implicants, AndOp):
                prime_implicants_ast = LogicalExpressionOracle._normalize_literal_indices(
                    prime_implicants.to_ast(), prime_implicants.usupport
                )
                sols = [[l[1] for l in prime_implicants_ast[1:]]]
            elif isinstance(prime_implicants, OrOp):
                expr_complete_sum = self._expr.complete_sum()
                complete_sum_ast = LogicalExpressionOracle._normalize_literal_indices(
                    expr_complete_sum.to_ast(), expr_complete_sum.usupport
                )
                sols = [[c[1]] if c[0] == 'lit' else [l[1] for l in c[1:]] for c in complete_sum_ast[1:]]
            elif isinstance(prime_implicants, Variable):
                sols = [[prime_implicants.to_ast()[1]]]
            else:
                raise AquaError('Unexpected solution: {}'.format(prime_implicants))
            for sol in sols:
                if set(sol).issubset(assignment):
                    return True, assignment
            else:
                return False, assignment
