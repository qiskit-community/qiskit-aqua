# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
The Boolean Logic Utility Classes.
"""

import logging
from abc import abstractmethod, ABC

from dlx import DLX
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


def get_prime_implicants(ones=None, dcs=None):
    """
    Compute all prime implicants for a truth table using the Quine-McCluskey Algorithm

    Args:
        ones (list of int): The list of integers corresponding to '1' outputs
        dcs (list of int): The list of integers corresponding to don't-cares

    Return:
        list of lists of int, representing all prime implicants
    """

    def combine_terms(terms, num1s_dict=None):
        if num1s_dict is None:
            num1s_dict = {}
            for num in terms:
                num1s = bin(num).count('1')
                if not num1s in num1s_dict:
                    num1s_dict[num1s] = [num]
                else:
                    num1s_dict[num1s].append(num)

        new_implicants = {}
        new_num1s_dict = {}
        prime_dict = {mt: True for mt in sorted(terms)}
        cur_num1s, max_num1s = min(num1s_dict.keys()), max(num1s_dict.keys())
        while cur_num1s < max_num1s:
            if cur_num1s in num1s_dict and (cur_num1s + 1) in num1s_dict:
                for cur_term in sorted(num1s_dict[cur_num1s]):
                    for next_term in sorted(num1s_dict[cur_num1s + 1]):
                        if isinstance(cur_term, int):
                            diff_mask = dc_mask = cur_term ^ next_term
                            implicant_mask = cur_term & next_term
                        elif isinstance(cur_term, tuple):
                            if terms[cur_term][1] == terms[next_term][1]:
                                diff_mask = terms[cur_term][0] ^ terms[next_term][0]
                                dc_mask = diff_mask | terms[cur_term][1]
                                implicant_mask = terms[cur_term][0] & terms[next_term][0]
                            else:
                                continue
                        else:
                            raise AquaError('Unexpected type: {}.'.format(type(cur_term)))
                        if bin(diff_mask).count('1') == 1:
                            prime_dict[cur_term] = False
                            prime_dict[next_term] = False
                            if isinstance(cur_term, int):
                                cur_implicant = (cur_term, next_term)
                            elif isinstance(cur_term, tuple):
                                cur_implicant = tuple(sorted((*cur_term, *next_term)))
                            else:
                                raise AquaError('Unexpected type: {}.'.format(type(cur_term)))
                            new_implicants[cur_implicant] = (
                                implicant_mask,
                                dc_mask
                            )
                            num1s = bin(implicant_mask).count('1')
                            if not num1s in new_num1s_dict:
                                new_num1s_dict[num1s] = [cur_implicant]
                            else:
                                if not cur_implicant in new_num1s_dict[num1s]:
                                    new_num1s_dict[num1s].append(cur_implicant)
            cur_num1s += 1
        return new_implicants, new_num1s_dict, prime_dict

    terms = ones + dcs
    cur_num1s_dict = None

    prime_implicants = []

    while True:
        next_implicants, next_num1s_dict, cur_prime_dict = combine_terms(terms, num1s_dict=cur_num1s_dict)
        for implicant in cur_prime_dict:
            if cur_prime_dict[implicant]:
                if isinstance(implicant, int):
                    if implicant not in dcs:
                        prime_implicants.append((implicant,))
                else:
                    if not set.issubset(set(implicant), dcs):
                        prime_implicants.append(implicant)
        if next_implicants:
            terms = next_implicants
            cur_num1s_dict = next_num1s_dict
        else:
            break

    return prime_implicants


def get_exact_covers(cols, rows, num_cols=None):
    """
    Use Algorithm X to get all solutions to the exact cover problem

    https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X

    Args:
          cols (list of int): A list of integers representing the columns to be covered
          rows (list of list of int): A list of lists of integers representing the rows
          num_cols (int): The total number of columns

    Returns:
        All exact covers
    """
    if num_cols is None:
        num_cols = max(cols) + 1
    ec = DLX([(c, 0 if c in cols else 1) for c in range(num_cols)])
    ec.appendRows([[c] for c in cols])
    ec.appendRows(rows)
    exact_covers = []
    for s in ec.solve():
        cover = []
        for i in s:
            cover.append(ec.getRowList(i))
        exact_covers.append(cover)
    return exact_covers


def _check_variables(vs):
    _vs = []
    for v in vs:
        if v in _vs:
            continue
        elif -v in _vs:
            return None
        else:
            _vs.append(v)
    return sorted(_vs, key=abs)


def logic_or(signed_vars, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
    """
    Build a collective disjunction (OR) circuit in place using mct.

    Args:
        signed_vars ([int]): The desired disjunctive clause as represented by a list of non-zero integers,
            whose absolute values indicate the variables, where negative signs correspond to negations.
        circuit (QuantumCircuit): The QuantumCircuit object to build the disjunction on.
        variable_register (QuantumRegister): The QuantumRegister holding the variable qubits. Note that the
            qubit indices are 0-based, so `variable_register[i]` correspond to variable `i-1` in `clause_expr`.
        target_qubit (tuple(QuantumRegister, int)): The target qubit to hold the disjunction result.
        ancillary_register (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """

    signed_vars = _check_variables(signed_vars)
    circuit.u3(pi, 0, pi, target_qubit)
    if signed_vars is not None:
        qs = [abs(v) for v in signed_vars]
        ctl_bits = [variable_register[idx - 1] for idx in qs]
        anc_bits = [ancillary_register[idx] for idx in range(len(qs) - 2)] if ancillary_register else None
        for idx in [v for v in signed_vars if v > 0]:
            circuit.u3(pi, 0, pi, variable_register[idx - 1])
        circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
        for idx in [v for v in signed_vars if v > 0]:
            circuit.u3(pi, 0, pi, variable_register[idx - 1])


def logic_and(signed_vars, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
    """
    Build a collective conjunction (AND) circuit in place using mct.

    Args:
        signed_vars ([int]): The desired disjunctive clause as represented by a list of non-zero integers,
            whose absolute values indicate the variables, where negative signs correspond to negations.
        circuit (QuantumCircuit): The QuantumCircuit object to build the conjunction on.
        variable_register (QuantumRegister): The QuantumRegister holding the variable qubits. Note that the
            qubit indices are 0-based, so `variable_register[i]` correspond to variable `i-1` in `clause_expr`.
        target_qubit (tuple(QuantumRegister, int)): The target qubit to hold the conjunction result.
        ancillary_register (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """

    signed_vars = _check_variables(signed_vars)
    if signed_vars is not None:
        qs = [abs(v) for v in signed_vars]
        ctl_bits = [variable_register[idx - 1] for idx in qs]
        anc_bits = [ancillary_register[idx] for idx in range(len(qs) - 2)] if ancillary_register else None
        for idx in [v for v in signed_vars if v < 0]:
            circuit.u3(pi, 0, pi, variable_register[-idx - 1])
        circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
        for idx in [v for v in signed_vars if v < 0]:
            circuit.u3(pi, 0, pi, variable_register[-idx - 1])


class BooleanLogicNormalForm(ABC):

    @staticmethod
    def _get_ast_depth(ast):
        if ast[0] == 'const' or ast[0] == 'lit':
            return 0
        else:
            return 1 + max([BooleanLogicNormalForm._get_ast_depth(c) for c in ast[1:]])

    @staticmethod
    def _get_ast_num_vars(ast):
        if ast[0] == 'const':
            return 0

        all_vars = set()

        def get_ast_vars(cur_ast):
            if cur_ast[0] == 'lit':
                all_vars.add(abs(cur_ast[1]))
            else:
                for c in cur_ast[1:]:
                    get_ast_vars(c)

        get_ast_vars(ast)
        return max(all_vars)

    """
    The base abstract class for:
    - CNF (Conjunctive Normal Forms),
    - DNF (Disjunctive Normal Forms), and
    - ESOP (Exclusive Sum of Products)
    """
    def __init__(self, ast, num_vars=None):
        """
        Constructor.

        Args:
            ast (tuple): The logic expression as an Abstract Syntax Tree (AST) tuple
            num_vars (int): Number of boolean variables
        """

        ast_depth = BooleanLogicNormalForm._get_ast_depth(ast)

        if ast_depth > 2:
            raise AquaError('Expressions of depth greater than 2 are not supported.')
        self._depth = ast_depth
        inferred_num_vars = BooleanLogicNormalForm._get_ast_num_vars(ast)
        if num_vars is None:
            self._num_variables = inferred_num_vars
        else:
            if inferred_num_vars > num_vars:
                raise AquaError('{} variables present, but only {} specified.'.format(inferred_num_vars, num_vars))
            self._num_variables = num_vars

        if ast_depth == 0:
            self._ast = ast
            self._num_clauses = 0
            self._max_clause_size = 0
        else:
            if ast_depth == 1:
                if self._depth == 1:
                    self._num_clauses = 1
                    self._max_clause_size = len(ast) - 1
                    self._ast = ast
                else:  # depth == 2:
                    if ast[0] == 'and':
                        op = 'or'
                    elif ast[0] == 'or':
                        op = 'and'
                    else:
                        raise AquaError('Unexpected expression root operator {}.'.format(ast[0]))
                    self._num_clauses = len(ast) - 1
                    self._max_clause_size = 1
                    self._ast = (ast[0], *[(op, l) for l in ast[1:]])

            else:  # self._depth == 2
                self._num_clauses = len(ast) - 1
                self._max_clause_size = max([len(l) - 1 for l in ast[1:]])
                self._ast = ast

        self._variable_register = None
        self._clause_register = None
        self._output_register = None
        self._ancillary_register = None

    @property
    def num_variables(self):
        return self._num_variables

    @property
    def num_clauses(self):
        return self._num_clauses

    @property
    def variable_register(self):
        return self._variable_register

    @property
    def clause_register(self):
        return self._clause_register

    @property
    def output_register(self):
        return self._output_register

    @property
    def ancillary_register(self):
        return self._ancillary_register

    @staticmethod
    def _set_up_register(num_qubits_needed, provided_register, description):
        if provided_register == 'skip':
            return None
        else:
            if provided_register is None:
                if num_qubits_needed > 0:
                    return QuantumRegister(num_qubits_needed, name=description[0])
            else:
                num_qubits_provided = len(provided_register)
                if num_qubits_needed > num_qubits_provided:
                    raise ValueError(
                        'The {} QuantumRegister needs {} qubits, but the provided register contains only {}.'.format(
                            description, num_qubits_needed, num_qubits_provided
                        ))
                else:
                    return provided_register

    def _set_up_circuit(
            self,
            circuit=None,
            variable_register=None,
            clause_register=None,
            output_register=None,
            output_idx=None,
            ancillary_register=None,
            mct_mode='basic'
    ):
        self._variable_register = BooleanLogicNormalForm._set_up_register(
            self.num_variables, variable_register, 'variable'
        )
        if self._depth > 1:
            self._clause_register = BooleanLogicNormalForm._set_up_register(
                self.num_clauses, clause_register, 'clause'
            )
        self._output_register = BooleanLogicNormalForm._set_up_register(
            1, output_register, 'output'
        )
        self._output_idx = output_idx if output_idx else 0

        max_num_ancillae = max(
            max(
                self._num_clauses if self._clause_register else 0,
                self._max_clause_size
            ) - 2,
            0
        )
        num_ancillae = 0
        if mct_mode == 'basic':
            num_ancillae = max_num_ancillae
        elif mct_mode == 'advanced':
            if max_num_ancillae >= 3:
                num_ancillae = 1
        elif mct_mode == 'noancilla':
            pass
        else:
            raise ValueError('Unsupported MCT mode {}.'.format(mct_mode))

        self._ancillary_register = BooleanLogicNormalForm._set_up_register(
            num_ancillae, ancillary_register, 'ancilla'
        )

        if circuit is None:
            circuit = QuantumCircuit()
            if self._variable_register:
                circuit.add_register(self._variable_register)
            if self._clause_register:
                circuit.add_register(self._clause_register)
            if self._output_register:
                circuit.add_register(self._output_register)
            if self._ancillary_register:
                circuit.add_register(self._ancillary_register)
        return circuit

    def _construct_circuit_for_tiny_expr(self, circuit, output_idx=0):
        if self._ast == ('const', 1):
            circuit.u3(pi, 0, pi, self._output_register[output_idx])
        elif self._ast[0] == 'lit':
            idx = abs(self._ast[1]) - 1
            if self._ast[1] < 0:
                circuit.u3(pi, 0, pi, self._variable_register[idx])
            circuit.cx(self._variable_register[idx], self._output_register[output_idx])

    @abstractmethod
    def construct_circuit(self, *args, **kwargs):
        raise NotImplementedError


class CNF(BooleanLogicNormalForm):
    """
    Class for constructing circuits for Conjunctive Normal Forms
    """
    def construct_circuit(
            self,
            circuit=None,
            variable_register=None,
            clause_register=None,
            output_register=None,
            ancillary_register=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum register to use for problem variables
            clause_register (QuantumRegister): The optional quantum register to use for problem clauses
            output_register (QuantumRegister): The optional quantum register to use for holding the output
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            variable_register=variable_register,
            clause_register=clause_register,
            output_register=output_register,
            ancillary_register=ancillary_register,
            mct_mode=mct_mode
        )

        if self._depth == 0:
            self._construct_circuit_for_tiny_expr(circuit)
        elif self._depth == 1:
            lits = [l[1] for l in self._ast[1:]]
            logic_and(
                lits,
                circuit,
                self._variable_register,
                self._output_register[0],
                self._ancillary_register,
                mct_mode
            )
        else:  # self._depth == 2:
            # compute all clauses
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'or':
                    lits = [l[1] for l in clause_expr[1:]]
                elif clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                else:
                    raise AquaError(
                        'Operator "{}" of clause {} in logic expression {} is unexpected.'.format(
                            clause_expr[0], clause_index, self._ast)
                    )
                logic_or(
                    lits,
                    circuit,
                    self._variable_register,
                    self._clause_register[clause_index],
                    self._ancillary_register,
                    mct_mode
                )

            # collect results from all clauses
            circuit.mct(
                self._clause_register,
                self._output_register[self._output_idx],
                self._ancillary_register,
                mode=mct_mode
            )

            # uncompute all clauses
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'or':
                    lits = [l[1] for l in clause_expr[1:]]
                else:  # clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                logic_or(
                    lits,
                    circuit,
                    self._variable_register,
                    self._clause_register[clause_index],
                    self._ancillary_register,
                    mct_mode
                )

        return circuit


class DNF(BooleanLogicNormalForm):
    """
    Class for constructing circuits for Disjunctive Normal Forms
    """
    def construct_circuit(
            self,
            circuit=None,
            variable_register=None,
            clause_register=None,
            output_register=None,
            ancillary_register=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum register to use for problem variables
            clause_register (QuantumRegister): The optional quantum register to use for problem clauses
            output_register (QuantumRegister): The optional quantum register to use for holding the output
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            variable_register=variable_register,
            clause_register=clause_register,
            output_register=output_register,
            ancillary_register=ancillary_register,
            mct_mode=mct_mode
        )
        if self._depth == 0:
            self._construct_circuit_for_tiny_expr(circuit)
        elif self._depth == 1:
            lits = [l[1] for l in self._ast[1:]]
            logic_or(
                lits,
                circuit,
                self._variable_register,
                self._output_register[0],
                self._ancillary_register,
                mct_mode
            )
        else:  # self._depth == 2
            # compute all clauses
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'and':
                    lits = [l[1] for l in clause_expr[1:]]
                elif clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                else:
                    raise AquaError(
                        'Operator "{}" of clause {} in logic expression {} is unexpected.'.format(
                            clause_expr[0], clause_index, self._ast)
                    )
                logic_and(
                    lits,
                    circuit,
                    self._variable_register,
                    self._clause_register[clause_index],
                    self._ancillary_register,
                    mct_mode
                )

            # init the output qubit to 1
            circuit.u3(pi, 0, pi, self._output_register[self._output_idx])

            # collect results from all clauses
            circuit.u3(pi, 0, pi, self._clause_register)
            circuit.mct(
                self._clause_register,
                self._output_register[self._output_idx],
                self._ancillary_register,
                mode=mct_mode
            )
            circuit.u3(pi, 0, pi, self._clause_register)

            # uncompute all clauses
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'and':
                    lits = [l[1] for l in clause_expr[1:]]
                else:  # clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                logic_and(
                    lits,
                    circuit,
                    self._variable_register,
                    self._clause_register[clause_index],
                    self._ancillary_register,
                    mct_mode
                )
        return circuit


class ESOP(BooleanLogicNormalForm):
    """
    Class for constructing circuits for Exclusive Sum of Products
    """
    def construct_circuit(
            self,
            circuit=None,
            variable_register=None,
            output_register=None,
            output_idx=None,
            ancillary_register=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum register to use for problem variables
            output_register (QuantumRegister): The optional quantum register to use for holding the output
            output_idx (int): The index of the output register to write to
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            variable_register=variable_register,
            clause_register='skip',
            output_register=output_register,
            output_idx=output_idx,
            ancillary_register=ancillary_register,
            mct_mode=mct_mode
        )

        # compute all clauses
        if self._depth == 0:
            self._construct_circuit_for_tiny_expr(circuit, output_idx=output_idx)
        else:
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'and':
                    lits = [l[1] for l in clause_expr[1:]]
                else:  # clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                logic_and(
                    lits,
                    circuit,
                    self._variable_register,
                    self._output_register[self._output_idx],
                    self._ancillary_register,
                    mct_mode
                )

        return circuit
