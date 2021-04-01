# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
The Quantum Dynamics algorithm.
"""

import logging

from typing import Optional, Union, Dict, Any
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.operators import LegacyBaseOperator
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.utils.validation import validate_min, validate_in_set

logger = logging.getLogger(__name__)


class EOH(QuantumAlgorithm):
    """
    The Quantum EOH (Evolution of Hamiltonian) algorithm.

    EOH provides the lower-level building blocks for simulating
    universal quantum systems. For any given quantum system that can be
    decomposed into local interactions (for example, a global hamiltonian as
    the weighted sum of several Pauli spin operators), the local
    interactions can then be used to approximate the global quantum system
    via, for example, Lloyd’s method or Trotter-Suzuki decomposition.
    """

    def __init__(self, operator: LegacyBaseOperator,
                 initial_state: Union[InitialState, QuantumCircuit],
                 evo_operator: LegacyBaseOperator,
                 evo_time: float = 1,
                 num_time_slices: int = 1,
                 expansion_mode: str = 'trotter',
                 expansion_order: int = 1,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args:
            operator: Operator to evaluate
            initial_state: Initial state for evolution
            evo_operator: Operator to evolve
            evo_time: Evolution time, has min value of 0
            num_time_slices: Number of time slices, has minimum value of 1
            expansion_mode: Either ``"trotter"`` (Lloyd's method) or ``"suzuki"``
                (for Trotter-Suzuki expansion)
            expansion_order: The Trotter-Suzuki expansion order.
            quantum_instance: Quantum Instance or Backend
        """
        validate_min('evo_time', evo_time, 0)
        validate_min('num_time_slices', num_time_slices, 1)
        validate_in_set('expansion_mode', expansion_mode, {'trotter', 'suzuki'})
        validate_min('expansion_order', expansion_order, 1)
        super().__init__(quantum_instance)
        self._operator = op_converter.to_weighted_pauli_operator(operator)
        self._initial_state = initial_state
        self._evo_operator = op_converter.to_weighted_pauli_operator(evo_operator)
        self._evo_time = evo_time
        self._num_time_slices = num_time_slices
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._ret = {}  # type: Dict[str, Any]

    def construct_circuit(self):
        """
        Construct the circuit.

        Returns:
            QuantumCircuit: the circuit.
        """
        quantum_registers = QuantumRegister(self._operator.num_qubits, name='q')
        if isinstance(self._initial_state, QuantumCircuit):
            qc = QuantumCircuit(quantum_registers)
            qc.compose(self._initial_state, inplace=True)
        else:
            qc = self._initial_state.construct_circuit('circuit', quantum_registers)

        qc.append(self._evo_operator.evolve(
            evo_time=self._evo_time,
            num_time_slices=self._num_time_slices,
            quantum_registers=quantum_registers,
            expansion_mode=self._expansion_mode,
            expansion_order=self._expansion_order,
        ), qc.qubits)

        return qc

    def _run(self):
        qc = self.construct_circuit()
        qc_with_op = self._operator.construct_evaluation_circuit(
            wave_function=qc, statevector_mode=self._quantum_instance.is_statevector)
        result = self._quantum_instance.execute(qc_with_op)
        self._ret['avg'], self._ret['std_dev'] = self._operator.evaluate_with_result(
            result=result, statevector_mode=self._quantum_instance.is_statevector)
        return self._ret
