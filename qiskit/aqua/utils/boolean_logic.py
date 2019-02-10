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


from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from .mct import mct

logger = logging.getLogger(__name__)


def _or(clause_expr, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
    qs = [abs(v) for v in clause_expr]
    ctl_bits = [variable_register[idx - 1] for idx in qs]
    anc_bits = [ancillary_register[idx] for idx in range(len(qs) - 2)] if ancillary_register else None
    for idx in [v for v in clause_expr if v > 0]:
        circuit.u3(pi, 0, pi, variable_register[idx - 1])
    circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
    for idx in [v for v in clause_expr if v > 0]:
        circuit.u3(pi, 0, pi, variable_register[idx - 1])


def _and(clause_expr, circuit, variable_register, target_qubit, ancillary_register, mct_mode):
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
            expr ([list]): List of lists of non-zero integers, where
                - each integer's absolute value indicates its variable index,
                - any negative sign indicates the negation for the corresponding variable,
                - each inner list corresponds to each clause of the logic expression, and
                - the outermost logic operation depends on the actual subclass (CNF, DNF, or ESOP)
        """

        self._expr = expr
        self._num_variables = max(set([abs(v) for v in list(itertools.chain.from_iterable(self._expr))]))
        self._num_clauses = len(self._expr)
        self._qr_variable = None
        self._qr_clause = None
        self._qr_outcome = None
        self._qr_ancilla = None

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
    def qr_variable(self):
        return self._qr_variable

    @property
    def qr_clause(self):
        return self._qr_clause

    @property
    def qr_outcome(self):
        return self._qr_outcome

    @property
    def qr_ancilla(self):
        return self._qr_ancilla

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
                    if num_qubits_needed < num_qubits_provided:
                        logger.warning(
                            'The {} QuantumRegister only needs {} qubits, but the provided register contains {}.'.format(
                                description, num_qubits_needed, num_qubits_provided
                            ))
                    return provided_register

    def _set_up_circuit(
            self,
            circuit=None,
            qr_variable=None,
            qr_clause=None,
            qr_outcome=None,
            qr_ancilla=None,
            mct_mode='basic'
    ):
        self._qr_variable = BooleanLogicNormalForm._set_up_register(self.num_variables, qr_variable, 'variable')
        self._qr_clause = BooleanLogicNormalForm._set_up_register(self.num_clauses, qr_clause, 'clause')
        self._qr_outcome = BooleanLogicNormalForm._set_up_register(1, qr_outcome, 'outcome')

        max_num_ancillae = max(max(self._num_clauses if self._qr_clause else 0, self._num_variables) - 2, 0)
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

        self._qr_ancilla = BooleanLogicNormalForm._set_up_register(num_ancillae, qr_ancilla, 'ancilla')

        if circuit is None:
            circuit = QuantumCircuit()
            if self._qr_variable:
                circuit.add_register(self._qr_variable)
            if self._qr_clause:
                circuit.add_register(self._qr_clause)
            if self._qr_outcome:
                circuit.add_register(self._qr_outcome)
            if self._qr_ancilla:
                circuit.add_register(self._qr_ancilla)
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
            qr_variable=None,
            qr_clause=None,
            qr_outcome=None,
            qr_ancilla=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            qr_variable (QuantumRegister): The optional quantum register to use for problem variables
            qr_clause (QuantumRegister): The optional quantum register to use for problem clauses
            qr_outcome (QuantumRegister): The optional quantum register to use for holding the output
            qr_ancilla (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            qr_variable=qr_variable,
            qr_clause=qr_clause,
            qr_outcome=qr_outcome,
            qr_ancilla=qr_ancilla,
            mct_mode=mct_mode
        )

        # init all clause qubits to 1
        circuit.u3(pi, 0, pi, self._qr_clause)

        # compute all clauses
        for clause_index, clause_expr in enumerate(self._expr):
            _or(clause_expr, circuit, self._qr_variable, self._qr_clause[clause_index], self._qr_ancilla, mct_mode)

        # collect results from all clauses
        circuit.mct(
            [self._qr_clause[i] for i in range(len(self._qr_clause))],
            self._qr_outcome[0],
            [self._qr_ancilla[i] for i in range(len(self._qr_ancilla))] if self._qr_ancilla else [],
            mode=mct_mode
        )

        # uncompute all clauses
        for clause_index, clause_expr in reversed(list(enumerate(self._expr))):
            _or(clause_expr, circuit, self._qr_variable, self._qr_clause[clause_index], self._qr_ancilla, mct_mode)

        # reset all clause qubits to 0
        circuit.u3(pi, 0, pi, self._qr_clause)

        return circuit


class DNF(BooleanLogicNormalForm):
    """
    Class for constructing circuits for Disjunctive Normal Forms
    """
    def construct_circuit(
            self,
            circuit=None,
            qr_variable=None,
            qr_clause=None,
            qr_outcome=None,
            qr_ancilla=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            qr_variable (QuantumRegister): The optional quantum register to use for problem variables
            qr_clause (QuantumRegister): The optional quantum register to use for problem clauses
            qr_outcome (QuantumRegister): The optional quantum register to use for holding the output
            qr_ancilla (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            qr_variable=qr_variable,
            qr_clause=qr_clause,
            qr_outcome=qr_outcome,
            qr_ancilla=qr_ancilla,
            mct_mode=mct_mode
        )

        # compute all clauses
        for clause_index, clause_expr in enumerate(self._expr):
            _and(clause_expr, circuit, self._qr_variable, self._qr_clause[clause_index], self._qr_ancilla, mct_mode)

        # init the outcome qubit to 1
        circuit.u3(pi, 0, pi, self._qr_outcome)

        # collect results from all clauses
        circuit.u3(pi, 0, pi, self._qr_clause)
        circuit.mct(
            [self._qr_clause[i] for i in range(len(self._qr_clause))],
            self._qr_outcome[0],
            [self._qr_ancilla[i] for i in range(len(self._qr_ancilla))] if self._qr_ancilla else [],
            mode=mct_mode
        )
        circuit.u3(pi, 0, pi, self._qr_clause)

        # uncompute all clauses
        for clause_index, clause_expr in reversed(list(enumerate(self._expr))):
            _and(clause_expr, circuit, self._qr_variable, self._qr_clause[clause_index], self._qr_ancilla, mct_mode)

        return circuit


class ESOP(BooleanLogicNormalForm):
    """
    Class for constructing circuits for Exclusive Sum of Products
    """
    def construct_circuit(
            self,
            circuit=None,
            qr_variable=None,
            qr_outcome=None,
            qr_ancilla=None,
            mct_mode='basic'
    ):
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            qr_variable (QuantumRegister): The optional quantum register to use for problem variables
            qr_outcome (QuantumRegister): The optional quantum register to use for holding the output
            qr_ancilla (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        """

        circuit = self._set_up_circuit(
            circuit=circuit,
            qr_variable=qr_variable,
            qr_clause='skip',
            qr_outcome=qr_outcome,
            qr_ancilla=qr_ancilla,
            mct_mode=mct_mode
        )

        # compute all clauses
        for clause_index, clause_expr in enumerate(self._expr):
            _and(clause_expr, circuit, self._qr_variable, self._qr_outcome[0], self._qr_ancilla, mct_mode)

        return circuit
