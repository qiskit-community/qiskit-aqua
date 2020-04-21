# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm Base """

import logging
from typing import Union

from qiskit.providers import BaseBackend

from qiskit.aqua import QuantumInstance
from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..combo_operators import ListOp, SummedOp
from ..primitive_operators import PauliOp
from ..state_functions import StateFn, CircuitStateFn, OperatorStateFn

logger = logging.getLogger(__name__)


class AerPauliExpectation(ExpectationBase):
    """ An Expectation Value algorithm for using Aer's operator snapshot to
    take expectations of quantum state circuits over Pauli observables.

    """

    def __init__(self,
                 operator: OperatorBase = None,
                 state: OperatorBase = None,
                 backend: BaseBackend = None):
        """
        Args:

        """
        super().__init__()
        self._operator = operator
        self._state = state
        self.backend = backend
        self._snapshot_op = None

    # TODO setters which wipe state

    @property
    def operator(self) -> OperatorBase:
        return self._operator

    @operator.setter
    def operator(self, operator: OperatorBase) -> None:
        self._operator = operator
        self._snapshot_op = None

    @property
    def state(self) -> OperatorBase:
        """ returns state """
        return self._state

    @state.setter
    def state(self, state: OperatorBase) -> None:
        self._state = state
        self._snapshot_op = None

    @property
    def quantum_instance(self) -> QuantumInstance:
        """ returns quantum instance """
        return self._circuit_sampler.quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance) -> None:
        self._circuit_sampler.quantum_instance = quantum_instance

    # TODO refactor to just rely on this
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Accept an Operator and return a new Operator with the Pauli measurements replaced by
        AerSnapshot-based expectation circuits. """
        if isinstance(operator, OperatorStateFn) and operator.is_measurement:
            return self._replace_pauli_sums(operator.primitive) * operator.coeff
        elif isinstance(operator, ListOp):
            return operator.traverse(self.convert)
        else:
            return operator

    # pylint: disable=inconsistent-return-statements,import-outside-toplevel
    @classmethod
    def _replace_pauli_sums(cls, operator):
        from qiskit.providers.aer.extensions import SnapshotExpectationValue
        if isinstance(operator, SummedOp):
            paulis = [[meas.coeff, meas.primitive] for meas in operator.oplist]
            snapshot_instruction = SnapshotExpectationValue('expval', paulis, variance=True)
            snapshot_op = CircuitStateFn(snapshot_instruction, is_measurement=True)
            return snapshot_op
        if isinstance(operator, PauliOp):
            paulis = [[operator.coeff, operator.primitive]]
            snapshot_instruction = SnapshotExpectationValue('expval', paulis, variance=True)
            snapshot_op = CircuitStateFn(snapshot_instruction, is_measurement=True)
            return snapshot_op
        if isinstance(operator, ListOp):
            return operator.traverse(cls._replace_pauli_sums)

    def expectation_op(self) -> OperatorBase:
        """ expectation op """

        snapshot_meas = self._replace_pauli_sums(self._operator)
        return snapshot_meas

    def compute_expectation(self,
                            state: OperatorBase = None,
                            params: dict = None) -> Union[float, complex, OperatorBase]:
        # Wipes caches in setter
        if state and not state == self.state:
            self.state = state

        if 'QuantumCircuit' in self.state.primitive_strings():
            if not self._snapshot_op:
                snapshot_meas = self.expectation_op()
                self._snapshot_op = snapshot_meas.compose(self.state).reduce()

            measured_op = self._circuit_sampler.convert(self._snapshot_op, params=params)
            # TODO once https://github.com/Qiskit/qiskit-aer/pull/485 goes through
            # self._quantum_instance._run_config.parameterizations = ...
            # result = self.quantum_instance.execute(list(self._snapshot_circuit.values()))
            return measured_op.eval()
        else:
            # If no circuits to run (i.e. state is a Dict, eval directly)
            return StateFn(self._operator, is_measurement=True).eval(self.state)

    def compute_standard_deviation(self,
                                   state: OperatorBase = None,
                                   params: dict = None) -> float:
        """ compute standard deviation """
        return 0.0
