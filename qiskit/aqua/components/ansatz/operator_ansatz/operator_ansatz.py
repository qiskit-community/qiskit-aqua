# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM  2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TODO"""

from typing import Union, List

from qiskit.circuit import ParameterVector
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.ansatz import Ansatz
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator


class OperatorAnsatz(Ansatz):
    """TODO"""

    def __init__(self, num_qubits,
                 operators: Union[BaseOperator, List[BaseOperator]],
                 parameter_prefix: str = '_',
                 insert_barriers: bool = False) -> None:
        """TODO"""

        super().__init__(insert_barriers=insert_barriers)
        self._num_qubits = num_qubits
        if isinstance(operators, BaseOperator):
            operators = [operators]

        params = ParameterVector(parameter_prefix, length=len(operators))
        for param, operator in zip(params, operators):
            self.append(self._get_evolution_layer(operator, param))

    def _get_evolution_layer(self, operator, param):
        """TODO"""
        pauli_op = to_weighted_pauli_operator(operator)
        return pauli_op.evolve_instruction(evo_time=param)
