# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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
from sympy.logic import simplify_logic
from sympy.logic.boolalg import is_cnf, is_dnf, BooleanTrue, BooleanFalse
from qiskit import QuantumCircuit, QuantumRegister

from qiskit.aqua import AquaError
from qiskit.aqua.circuits import CNF, DNF
from qiskit.aqua.utils.validation import validate_in_set
from .oracle import Oracle
from .ast_utils import get_ast

logger = logging.getLogger(__name__)


class LogicalExpressionOracle(Oracle):
    r"""
    The Logical Expression Quantum Oracle.

    The Logical Expression Oracle, as its name suggests, constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (`AND`), `|` (`OR`), `~` (`NOT`), and `^` (`XOR`),
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) ^ (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals.

    The following is an example of a CNF expressed in DIMACS format:

    .. code:: text

      c This is an example DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
      p cnf 3 5
      -1 -2 -3 0
      1 -2 3 0
      1 2 -3 0
      1 -2 -3 0
      -1 2 3 0

    The first line, following the `c` character, is a comment. The second line specifies that the
    CNF is over three boolean variables --- let us call them  :math:`x_1, x_2, x_3`, and contains
    five clauses.  The five clauses, listed afterwards, are implicitly joined by the logical `AND`
    operator, :math:`\land`, while the variables in each clause, represented by their indices,
    are implicitly disjoined by the logical `OR` operator, :math:`lor`. The :math:`-` symbol
    preceding a boolean variable index corresponds to the logical `NOT` operator, :math:`lnot`.
    Character `0` (zero) marks the end of each clause.  Essentially, the code above corresponds
    to the following CNF:

    :math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3)
    \land (x_1 \lor \lnot x_2 \lor x_3)
    \land (x_1 \lor x_2 \lor \lnot x_3)
    \land (x_1 \lor \lnot x_2 \lor \lnot x_3)
    \land (\lnot x_1 \lor x_2 \lor x_3)`.

    This is an example showing how to search for a satisfying assignment to an SAT problem encoded
    in DIMACS using the `Logical Expression oracle with the Grover algorithm.
    <https://github.com/Qiskit/qiskit-tutorials-community/blob/master/optimization/grover.ipynb>`__

    Logic expressions, regardless of the input formats, are parsed and stored as Abstract Syntax
    Tree (AST) tuples, from which the corresponding circuits are constructed. The oracle circuits
    can then be used with any oracle-oriented algorithms when appropriate. For example, an oracle
    built from a DIMACS input can be used with the Grover's algorithm to search for a satisfying
    assignment to the encoded SAT instance.

    By default, the Logical Expression oracle will not try to apply any optimization when building
    the circuits. For any DIMACS input, the constructed circuit truthfully recreates each inner
    disjunctive clauses as well as the outermost conjunction; For other arbitrary input expression,
    It only tries to convert it to a CNF or DNF (Disjunctive Normal Form, similar to CNF, but with
    inner conjunctions and a outer disjunction) before constructing its circuit. This, for example,
    could be good for educational purposes, where a user would like to compare a built circuit
    against their input expression to examine and analyze details. However, this often leads
    to relatively deep circuits that possibly also involve many ancillary qubits. The oracle
    therefore, provides the option to try to optimize the input logical expression before
    building its circuit.
    """

    def __init__(self,
                 expression: str,
                 optimization: bool = False,
                 mct_mode: str = 'basic') -> None:
        """
        Args:
            expression: The string of the desired logical expression.
                It could be either in the DIMACS CNF format,
                or a general boolean logical expression, such as 'a ^ b' and 'v[0] & (~v[1] | v[2])'
            optimization: Boolean flag for attempting logical expression optimization
            mct_mode: The mode to use for building Multiple-Control Toffoli.
        Raises:
            AquaError: Invalid input
        """
        validate_in_set('mct_mode', mct_mode,
                        {'basic', 'basic-dirty-ancilla',
                         'advanced', 'noancilla'})
        super().__init__()

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
            except Exception as ex:
                raise AquaError(
                    'Failed to parse the input expression: {}.'.format(orig_expression)) from ex
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

        if self._optimization or (not is_cnf(self._expr) and not is_dnf(self._expr)):
            expr = simplify_logic(self._expr)
        else:
            expr = self._expr

        if isinstance(expr, BooleanTrue):
            ast = 'const', 1
        elif isinstance(expr, BooleanFalse):
            ast = 'const', 0
        else:
            ast = get_ast(self._var_to_lit, expr)

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
