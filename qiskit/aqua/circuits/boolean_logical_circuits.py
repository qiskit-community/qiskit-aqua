# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Boolean Logical DNF, CNF, and ESOP Circuits.
"""

import logging
from abc import abstractmethod, ABC

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import AND, OR

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class BooleanLogicNormalForm(ABC):
    """
    Boolean Logical DNF, CNF, and ESOP Circuits.
    The base abstract class for:
    - CNF (Conjunctive Normal Forms),
    - DNF (Disjunctive Normal Forms), and
    - ESOP (Exclusive Sum of Products)
    """
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

    @staticmethod
    def _lits_to_flags(vs):
        _vs = []
        for v in vs:
            if v in _vs:
                continue
            if -v in _vs:
                return None
            else:
                _vs.append(v)
        flags = abs(max(_vs, key=abs)) * [0]
        for v in _vs:
            flags[abs(v) - 1] = 1 if v > 0 else -1
        return flags

    def __init__(self, ast, num_vars=None):
        """
        Constructor.

        Args:
            ast (tuple): The logic expression as an Abstract Syntax Tree (AST) tuple
            num_vars (int): Number of boolean variables
        Raises:
            AquaError: invalid input
        """
        self._output_idx = None
        ast_depth = BooleanLogicNormalForm._get_ast_depth(ast)

        if ast_depth > 2:
            raise AquaError('Expressions of depth greater than 2 are not supported.')
        self._depth = ast_depth
        inferred_num_vars = BooleanLogicNormalForm._get_ast_num_vars(ast)
        if num_vars is None:
            self._num_variables = inferred_num_vars
        else:
            if inferred_num_vars > num_vars:
                raise AquaError('{} variables present, but only {} '
                                'specified.'.format(inferred_num_vars, num_vars))
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
        """ return num variables """
        return self._num_variables

    @property
    def num_clauses(self):
        """ returns num clauses """
        return self._num_clauses

    @property
    def variable_register(self):
        """ returns variable register """
        return self._variable_register

    @property
    def clause_register(self):
        """ returns clause register """
        return self._clause_register

    @property
    def output_register(self):
        """ returns output register """
        return self._output_register

    @property
    def ancillary_register(self):
        """ returns ancillary register """
        return self._ancillary_register

    @staticmethod
    def _set_up_register(num_qubits_needed, provided_register, description):
        if isinstance(provided_register, str) and provided_register == 'skip':
            return None
        else:
            if provided_register is None:
                if num_qubits_needed > 0:
                    return QuantumRegister(num_qubits_needed, name=description[0])
            else:
                num_qubits_provided = len(provided_register)
                if num_qubits_needed > num_qubits_provided:
                    raise ValueError(
                        'The {} QuantumRegister needs {} qubits, '
                        'but the provided register contains only {}.'.format(
                            description, num_qubits_needed, num_qubits_provided))

                return provided_register

        return None

    def compute_num_ancillae(self, mct_mode='basic'):
        """ returns the number of ancillary qubits needed """
        max_num_ancillae = max(
            max(
                self._num_clauses if self._clause_register else 0,
                self._max_clause_size
            ) - 2,
            0
        )
        num_ancillae = 0
        if mct_mode in ('basic', 'basic-dirty-ancilla'):
            num_ancillae = max_num_ancillae
        elif mct_mode == 'advanced':
            if max_num_ancillae >= 3:
                num_ancillae = 1
        elif mct_mode == 'noancilla':
            pass
        else:
            raise ValueError('Unsupported MCT mode {}.'.format(mct_mode))
        return num_ancillae

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

        num_ancillae = self.compute_num_ancillae(mct_mode)

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
        if self._ast == ('const', 0):
            pass
        elif self._ast == ('const', 1):
            circuit.x(self._output_register[output_idx])
        elif self._ast[0] == 'lit':
            idx = abs(self._ast[1]) - 1
            if self._ast[1] < 0:
                circuit.x(self._variable_register[idx])
            circuit.cx(self._variable_register[idx], self._output_register[output_idx])
        else:
            raise AquaError('Unexpected tiny expression {}.'.format(self._ast))

    @abstractmethod
    def construct_circuit(self, *args, **kwargs):
        """ construct circuit """
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
    ):  # pylint: disable=arguments-differ
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum register
                        to use for problem variables
            clause_register (QuantumRegister): The optional quantum register
                        to use for problem clauses
            output_register (QuantumRegister): The optional quantum register
                        to use for holding the output
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        Raises:
            AquaError: invalid input
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
            flags = BooleanLogicNormalForm._lits_to_flags(lits)
            if flags is not None:
                and_circuit = AND(num_variable_qubits=len(self._variable_register),
                                  flags=flags, mcx_mode=mct_mode)
                qubits = self._variable_register[:] + [self._output_register[0]]
                if self._ancillary_register:
                    qubits += self._ancillary_register[:and_circuit.num_ancillas]

                circuit.compose(and_circuit, qubits, inplace=True)
        else:  # self._depth == 2:
            active_clause_indices = []
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
                flags = BooleanLogicNormalForm._lits_to_flags(lits)
                if flags is not None:
                    active_clause_indices.append(clause_index)
                    or_circuit = OR(num_variable_qubits=len(self._variable_register),
                                    flags=flags, mcx_mode=mct_mode)
                    qubits = self._variable_register[:] + [self._clause_register[clause_index]]
                    if self._ancillary_register:
                        qubits += self._ancillary_register[:or_circuit.num_ancillas]

                    circuit.compose(or_circuit, qubits, inplace=True)

            # collect results from all clauses
            circuit.mct(
                [self._clause_register[i] for i in active_clause_indices],
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
                flags = BooleanLogicNormalForm._lits_to_flags(lits)
                if flags is not None:
                    or_circuit = OR(num_variable_qubits=len(self._variable_register),
                                    flags=flags, mcx_mode=mct_mode)
                    qubits = self._variable_register[:] + [self._clause_register[clause_index]]
                    if self._ancillary_register:
                        qubits += self._ancillary_register[:or_circuit.num_ancillas]

                    circuit.compose(or_circuit, qubits, inplace=True)

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
    ):  # pylint: disable=arguments-differ
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum register
                            to use for problem variables
            clause_register (QuantumRegister): The optional quantum register
                            to use for problem clauses
            output_register (QuantumRegister): The optional quantum register
                            to use for holding the output
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        Raises:
            AquaError: invalid input
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
            flags = BooleanLogicNormalForm._lits_to_flags(lits)
            if flags is not None:
                or_circuit = OR(num_variable_qubits=len(self._variable_register),
                                flags=flags, mcx_mode=mct_mode)
                qubits = self._variable_register[:] + [self._output_register[0]]
                if self._ancillary_register:
                    qubits += self._ancillary_register[:or_circuit.num_ancillas]

                circuit.compose(or_circuit, qubits, inplace=True)
            else:
                circuit.x(self._output_register[0])
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
                flags = BooleanLogicNormalForm._lits_to_flags(lits)
                if flags is not None:
                    and_circuit = AND(num_variable_qubits=len(self._variable_register),
                                      flags=flags, mcx_mode=mct_mode)
                    qubits = self._variable_register[:] + [self._clause_register[clause_index]]
                    if self._ancillary_register:
                        qubits += self._ancillary_register[:and_circuit.num_ancillas]

                    circuit.compose(and_circuit, qubits, inplace=True)
                else:
                    circuit.x(self._clause_register[clause_index])

            # init the output qubit to 1
            circuit.x(self._output_register[self._output_idx])

            # collect results from all clauses
            circuit.x(self._clause_register)
            circuit.mct(
                self._clause_register,
                self._output_register[self._output_idx],
                self._ancillary_register,
                mode=mct_mode
            )
            circuit.x(self._clause_register)

            # uncompute all clauses
            for clause_index, clause_expr in enumerate(self._ast[1:]):
                if clause_expr[0] == 'and':
                    lits = [l[1] for l in clause_expr[1:]]
                elif clause_expr[0] == 'lit':
                    lits = [clause_expr[1]]
                flags = BooleanLogicNormalForm._lits_to_flags(lits)
                if flags is not None:
                    and_circuit = AND(num_variable_qubits=len(self._variable_register),
                                      flags=flags, mcx_mode=mct_mode)
                    qubits = self._variable_register[:] + [self._clause_register[clause_index]]
                    if self._ancillary_register:
                        qubits += self._ancillary_register[:and_circuit.num_ancillas]

                    circuit.compose(and_circuit, qubits, inplace=True)
                else:
                    circuit.x(self._clause_register[clause_index])
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
    ):  # pylint: disable=arguments-differ
        """
        Construct circuit.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from
            variable_register (QuantumRegister): The optional quantum
            register to use for problem variables
            output_register (QuantumRegister): The optional quantum
            register to use for holding the output
            output_idx (int): The index of the output register to write to
            ancillary_register (QuantumRegister): The optional quantum register to use as ancilla
            mct_mode (str): The mode to use for building Multiple-Control Toffoli

        Returns:
            QuantumCircuit: quantum circuit.
        Raises:
            AquaError: invalid input
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

        def build_clause(clause_expr):
            if clause_expr[0] == 'and':
                lits = [l[1] for l in clause_expr[1:]]
            elif clause_expr[0] == 'lit':
                lits = [clause_expr[1]]
            else:
                raise AquaError('Unexpected clause expression {}.'.format(clause_expr))
            flags = BooleanLogicNormalForm._lits_to_flags(lits)
            and_circuit = AND(num_variable_qubits=len(self._variable_register),
                              flags=flags, mcx_mode=mct_mode)
            qubits = self._variable_register[:] + [self._output_register[self._output_idx]]
            if self._ancillary_register:
                qubits += self._ancillary_register[:and_circuit.num_ancillas]

            circuit.compose(and_circuit, qubits, inplace=True)

        # compute all clauses
        if self._depth == 0:
            self._construct_circuit_for_tiny_expr(circuit, output_idx=output_idx)
        elif self._depth == 1:
            if self._ast[0] == 'xor':
                for cur_clause_expr in self._ast[1:]:
                    build_clause(cur_clause_expr)
            else:
                build_clause(self._ast)
        elif self._depth == 2:
            if not self._ast[0] == 'xor':
                raise AquaError('Unexpected root logical '
                                'operation {} for ESOP.'.format(self._ast[0]))
            for cur_clause_expr in self._ast[1:]:
                build_clause(cur_clause_expr)
        else:
            raise AquaError('Unexpected ESOP expression {}.'.format(self._ast))

        return circuit
