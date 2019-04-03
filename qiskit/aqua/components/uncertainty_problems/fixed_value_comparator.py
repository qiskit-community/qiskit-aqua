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
The Fixed Value Comparator.
"""

from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.circuits.gates import mct, logical_or

import numpy as np


class FixedValueComparator(CircuitFactory):
    """
    The Fixed Value Comparator.

    Operator compares basis states |i> against a given fixed level l and flips a target qubit if x >= l (or <= depending on parameters):
    |x>|0> --> |x>|1> if x >= l and |x>|0> otherwise.
    """

    def __init__(self, num_target_qubits, value, geq=True):
        super().__init__(num_target_qubits)
        self._value = value
        self._clauses = self._get_clauses(value)
        self._geq = geq

    @property
    def value(self):
        return self._value

    def required_ancillas(self):
        num_clause_toffoli_ancillas = 0
        for clause in self._clauses:
            num_clause_toffoli_ancillas = max(num_clause_toffoli_ancillas, len(clause) - 2)
        if len(self._clauses) > 1:
            num_clause_result_ancillas = len(self._clauses)
            num_or_ancillas = num_clause_result_ancillas + max(0, num_clause_result_ancillas - 2)
            return num_clause_toffoli_ancillas + num_clause_result_ancillas + num_or_ancillas
        else:
            return num_clause_toffoli_ancillas

    def required_ancillas_controlled(self):
        raise NotImplementedError()

    def _get_clauses(self, value):

        num_state_qubits = self.num_target_qubits - 1
        clauses = []
        if 0 < value < 2 ** num_state_qubits:

            n = int(np.ceil(np.log2(value)))
            for k in range(num_state_qubits - 1, n - 1, -1):
                clauses += [[k]]
            base_clause = []
            subtract = 0
            n_min = 1 + int(np.mod(value, 2) == 0)
            while n > n_min:
                base_clause += [(n-1)]

                subtract += 2 ** (n - 1)
                n_new = int(np.ceil(np.log2(value - subtract)))

                for k in range(n - 2, n_new - 1, -1):
                    new_clause = base_clause.copy()
                    new_clause += [k]
                    clauses += [new_clause]

                n = n_new

        return clauses

    def build(self, qc, q, q_ancillas=None, params=None):

        q_result = q[params['i_compare']]
        q_state = [q[i] for i in params['i_state']]
        num_state_qubits = self.num_target_qubits - 1

        uncompute = params.get('uncompute_ancillas', True)

        # evaluate clauses into ancillas
        num_clauses = len(self._clauses)
        if num_clauses > 1:

            # evaluate clauses on ancillas
            q_ancillas_ = [q_ancillas[i] for i in range(len(q_ancillas))]
            for k, clause in enumerate(self._clauses):
                q_controls = [q_state[i] for i in clause]
                qc.mct(q_controls, q_ancillas[k], q_ancillas_[num_clauses:])

            # apply OR to clause ancillas
            qc.OR(q_ancillas_[:num_clauses], q_result, q_ancillas_[num_clauses:])

            # uncompute clauses on ancillas
            if uncompute:
                for k, clause in enumerate(self._clauses):
                    q_controls = [q_state[i] for i in clause]
                    qc.mct(q_controls, q_ancillas[k], q_ancillas_[num_clauses:])

            if self._geq is False:
                qc.x(q_result)

        elif num_clauses == 1:
            q_controls = [q_state[i] for i in self._clauses[0]]
            qc.mct(q_controls, q_result, q_ancillas)

            if self._geq is False:
                qc.x(q_result)
        else:
            if self.value <= 0:
                if self._geq is True:
                    qc.x(q_result)
            elif self.value >= 2**num_state_qubits:
                if self._geq is False:
                    qc.x(q_result)
