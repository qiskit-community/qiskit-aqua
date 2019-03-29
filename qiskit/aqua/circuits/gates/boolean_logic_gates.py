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
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


def _logic_and(circuit, variable_register, flags, target_qubit, ancillary_register, mct_mode):
    if flags is not None:
        zvf = list(zip(variable_register, flags))
        ctl_bits = [v for v, f in zvf if f]
        anc_bits = [ancillary_register[idx] for idx in range(np.count_nonzero(flags) - 2)] if ancillary_register else None
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f < 0]
        circuit.mct(ctl_bits, target_qubit, anc_bits, mode=mct_mode)
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f < 0]


def _logic_or(circuit, qr_variables, flags, qb_target, qr_ancillae, mct_mode):
    circuit.u3(pi, 0, pi, qb_target)
    if flags is not None:
        zvf = list(zip(qr_variables, flags))
        ctl_bits = [v for v, f in zvf if f]
        anc_bits = [qr_ancillae[idx] for idx in range(np.count_nonzero(flags) - 2)] if qr_ancillae else None
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f > 0]
        circuit.mct(ctl_bits, qb_target, anc_bits, mode=mct_mode)
        [circuit.u3(pi, 0, pi, v) for v, f in zvf if f > 0]


def _do_checks(flags, qr_variables, qb_target, qr_ancillae, circuit):
    # check flags
    if flags is None:
        flags = [1 for i in range(len(qr_variables))]
    else:
        if len(flags) > len(qr_variables):
            raise AquaError('`flags` cannot be longer than `qr_variables`.')

    # check variables
    if isinstance(qr_variables, QuantumRegister):
        variable_qubits = [qb for qb, i in zip(qr_variables, flags) if not i == 0]
    else:
        raise ValueError('A QuantumRegister is expected for variables.')

    # check target
    if isinstance(qb_target, tuple):
        target_qubit = qb_target
    else:
        raise ValueError('A single qubit is expected for the target.')

    # check ancilla
    if qr_ancillae is None:
        ancillary_qubits = []
    elif isinstance(qr_ancillae, QuantumRegister):
        ancillary_qubits = [qb for qb in qr_ancillae]
    elif isinstance(qr_ancillae, list):
        ancillary_qubits = qr_ancillae
    else:
        raise ValueError('An optional list of qubits or a QuantumRegister is expected for ancillae.')

    all_qubits = variable_qubits + [target_qubit] + ancillary_qubits
    try:
        for qubit in all_qubits:
            circuit._check_qubit(qubit)
    except AttributeError as e: # TODO Temporary, _check_qubit may not exist
        logger.debug(str(e))

    circuit._check_dups(all_qubits)

    return flags


def logic_and(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='basic'):
    """
    Build a collective conjunction (AND) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the conjunction on.
        variable_register (QuantumRegister): The QuantumRegister holding the variable qubits.
        flags (list): A list of +1/-1/0 to mark negations or omissions of qubits.
        target_qubit (tuple(QuantumRegister, int)): The target qubit to hold the conjunction result.
        ancillary_register (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """
    flags = _do_checks(flags, qr_variables, qb_target, qr_ancillae, self)
    _logic_and(self, qr_variables, flags, qb_target, qr_ancillae, mct_mode)


def logic_or(self, qr_variables, qb_target, qr_ancillae, flags=None, mct_mode='basic'):
    """
    Build a collective disjunction (OR) circuit in place using mct.

    Args:
        self (QuantumCircuit): The QuantumCircuit object to build the disjunction on.
        qr_variables (QuantumRegister): The QuantumRegister holding the variable qubits.
        flags (list): A list of +1/-1/0 to mark negations or omissions of qubits.
        qb_target (tuple(QuantumRegister, int)): The target qubit to hold the disjunction result.
        qr_ancillae (QuantumRegister): The ancillary QuantumRegister for building the mct.
        mct_mode (str): The mct building mode.
    """
    flags = _do_checks(flags, qr_variables, qb_target, qr_ancillae, self)
    _logic_or(self, qr_variables, flags, qb_target, qr_ancillae, mct_mode)


QuantumCircuit.AND = logic_and
QuantumCircuit.OR = logic_or
