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

import itertools
import logging
from abc import abstractmethod, ABC

from dlx import DLX
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from qiskit.aqua import AquaError
from .mct import mct

logger = logging.getLogger(__name__)


def is_power_of_2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def get_prime_implicants(truth_table_values):
    """
    Compute all prime implicants for a truth table using the Quine-McCluskey Algorithm

    Args:
        truth_table_values (str): The bit string representing the truth table

    Return:
        A list of lists, representing all prime implicants
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

    if not is_power_of_2(len(truth_table_values)):
        raise AquaError('The truth table length needs to be a power of 2.')

    ones = [i for i, x in enumerate(truth_table_values) if x == '1']
    dc = [i for i, x in enumerate(truth_table_values) if x == 'x']
    candidates = ones + dc
    cur_num1s_dict = None

    prime_implicants = []

    while True:
        next_implicants, next_num1s_dict, cur_prime_dict = combine_terms(candidates, num1s_dict=cur_num1s_dict)
        for implicant in cur_prime_dict:
            if cur_prime_dict[implicant] and not set.issubset(set(implicant), dc):
                prime_implicants.append(implicant)
        if next_implicants:
            candidates = next_implicants
            cur_num1s_dict = next_num1s_dict
        else:
            break

    return prime_implicants


def get_all_exact_covers(cols, rows, col_range=None):
    """
    Use Algorithm X to get all solutions to the exact cover problem

    https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X

    Args:
          cols (list): A list of integers representing the columns to be covered
          rows (list of lists): A list of lists of integers representing the rows
          col_range (int): The upper range of the columns (i.e. max_col + 1)

    Returns:
        All exact covers
    """
    if col_range is None:
        col_range = max(cols) + 1
    ec = DLX([(c, 0 if c in cols else 1) for c in range(col_range)])
    ec.appendRows([[c] for c in cols])
    ec.appendRows(rows)
    all_covers = []
    for s in ec.solve():
        cover = []
        for i in s:
            cover.append(ec.getRowList(i))
        all_covers.append(cover)
    return all_covers


def logic_or(clause_expr, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
    clause_expr = sorted(clause_expr, key=abs)
    qs = [abs(v) for v in clause_expr]
    ctl_bits = [variable_register[idx - 1] for idx in qs]
    anc_bits = [ancillary_register[idx] for idx in range(len(qs) - 2)] if ancillary_register else None
    for idx in [v for v in clause_expr if v > 0]:
        circuit.u3(pi, 0, pi, variable_register[idx - 1])
    circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
    for idx in [v for v in clause_expr if v > 0]:
        circuit.u3(pi, 0, pi, variable_register[idx - 1])


def logic_and(clause_expr, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
    clause_expr = sorted(clause_expr, key=abs)
    qs = [abs(v) for v in clause_expr]
    ctl_bits = [variable_register[idx - 1] for idx in qs]
    anc_bits = [ancillary_register[idx] for idx in range(len(qs) - 2)] if ancillary_register else None
    for idx in [v for v in clause_expr if v < 0]:
        circuit.u3(pi, 0, pi, variable_register[-idx - 1])
    circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
    for idx in [v for v in clause_expr if v < 0]:
        circuit.u3(pi, 0, pi, variable_register[-idx - 1])


class BooleanLogicNormalForm(ABC):
    """
    The base abstract class for:
    - CNF (Conjunctive Normal Forms),
    - DNF (Disjunctive Normal Forms), and
    - ESOP (Exclusive Sum of Products)
    """
    def __init__(self, expr):
        """
        Constructor.

        Args:
            expr (list of lists of ints): List of lists of non-zero integers, where
                - each integer's absolute value indicates its variable index,
                - any negative sign indicates the negation for the corresponding variable,
                - each inner list corresponds to each clause of the logic expression, and
                - the outermost logic operation depends on the actual subclass (CNF, DNF, or ESOP)
        """

        self._expr = expr
        self._num_variables = max(set([abs(v) for v in list(itertools.chain.from_iterable(self._expr))]))
        self._num_clauses = len(self._expr)
        self._variable_register = None
        self._clause_register = None
        self._output_register = None
        self._ancillary_register = None

    @property
    def expr(self):
        return self._expr

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
                self._num_variables
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

        # init all clause qubits to 1
        circuit.u3(pi, 0, pi, self._clause_register)

        # compute all clauses
        for clause_index, clause_expr in enumerate(self._expr):
            logic_or(
                clause_expr,
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
        for clause_index, clause_expr in reversed(list(enumerate(self._expr))):
            logic_or(
                clause_expr,
                circuit,
                self._variable_register,
                self._clause_register[clause_index],
                self._ancillary_register,
                mct_mode
            )

        # reset all clause qubits to 0
        circuit.u3(pi, 0, pi, self._clause_register)

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

        # compute all clauses
        for clause_index, clause_expr in enumerate(self._expr):
            logic_and(
                clause_expr,
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
        for clause_index, clause_expr in reversed(list(enumerate(self._expr))):
            logic_and(
                clause_expr,
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
        for clause_index, clause_expr in enumerate(self._expr):
            logic_and(
                clause_expr,
                circuit,
                self._variable_register,
                self._output_register[self._output_idx],
                self._ancillary_register,
                mct_mode
            )

        return circuit
