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

from ..operator_base import OperatorBase
from .expectation_base import ExpectationBase
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..list_ops.summed_op import SummedOp
from ..primitive_ops.pauli_op import PauliOp
from ..state_fns.circuit_state_fn import CircuitStateFn
from ..state_fns.operator_state_fn import OperatorStateFn

logger = logging.getLogger(__name__)


class AerPauliExpectation(ExpectationBase):
    """ An Expectation Value algorithm for using Aer's operator snapshot to
    take expectations of quantum state circuits over Pauli observables.

    """

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
        # The 'expval_measurement' label on the snapshot instruction is special - the
        # CircuitSampler will look for it to know that the circuit is a Expectation
        # measurement, and not simply a
        # circuit to replace with a DictStateFn
        if isinstance(operator, SummedOp):
            paulis = [[meas.coeff, meas.primitive] for meas in operator.oplist]
            snapshot_instruction = SnapshotExpectationValue('expval_measurement',
                                                            paulis,
                                                            variance=True)
            snapshot_op = CircuitStateFn(snapshot_instruction, is_measurement=True)
            return snapshot_op
        if isinstance(operator, PauliOp):
            paulis = [[operator.coeff, operator.primitive]]
            snapshot_instruction = SnapshotExpectationValue('expval_measurement',
                                                            paulis,
                                                            variance=True)
            snapshot_op = CircuitStateFn(snapshot_instruction, is_measurement=True)
            return snapshot_op
        if isinstance(operator, ListOp):
            return operator.traverse(cls._replace_pauli_sums)

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        """ compute variance """

        # Need to do this to mimic Op structure
        def sum_variance(operator):
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])

        return sum_variance(exp_op)
