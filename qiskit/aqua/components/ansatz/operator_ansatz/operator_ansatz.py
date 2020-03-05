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

"""The operator ansatz."""

from typing import Union, List, Optional

from qiskit.circuit import Instruction, Parameter
from qiskit.tools import parallel_map

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.ansatz import Ansatz
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator


class OperatorAnsatz(Ansatz):
    """The operator ansatz.

    TODO Dedicated to UCCSD, QAOA etc.
    """

    def __init__(self,
                 operators: Union[BaseOperator, List[BaseOperator]],
                 qubit_indices: Optional[Union[List[int], List[List[int]]]] = None,
                 parameter_prefix: str = '_',
                 use_basis_gates: bool = False,
                 insert_barriers: bool = False) -> None:
        r"""Initialize the operator ansatz.

        Provided with a list of operators [\hat{O}_1, \hat{O}_2, \ldots], this Ansatz constructs
        an evolution instruction for each operator, parameterized on the evolution time.
        That is, the constructed Ansatz is

        .. math::
            e^{i\theta_1\hat{O}_1} e^{i\theta_2\hat{O}_2 \cdots}

        where [\theta_1, \theta_2, \ldots] are the parameters of the Ansatz.

        Args:
            operators: An operator (or list of) to specify the evolution instructions.
            qubit_indices: The indices where the evolution instruction should be inserted.
                Defaults to 0, ..., n - 1, where n is the number of qubits the operator acts on.
            parameter_prefix: The parameter prefix used for the default parameters.
                Defaults to '_'.
            use_basis_gates: If True, the basis gates (U1, U2, U3) are used in the evolution
                instruction, if False, the Pauli/Clifford gates are used.
            insert_barriers: Whether to insert barriers in between the evolution instructions.
        """
        # bring operators in the list format
        if isinstance(operators, BaseOperator):
            operators = [operators]

        self._use_basis_gates = use_basis_gates
        self._parameter_prefix = parameter_prefix

        params = [Parameter('{}{}'.format(self._parameter_prefix, i))
                  for i in range(len(operators))]
        evolution_layers = self._get_evolution_layers(operators, params)

        super().__init__(blocks=evolution_layers,
                         qubit_indices=qubit_indices,
                         insert_barriers=insert_barriers)

    def _get_single_evolution_layer(self, operator_param_pair):
        operator, param = operator_param_pair
        pauli_op = to_weighted_pauli_operator(operator)
        return pauli_op.evolve_instruction(evo_time=param,
                                           use_basis_gates=self._use_basis_gates)

    def _get_evolution_layers(self, operators: List[BaseOperator],
                              params: Union[List[float], List[Parameter]]) -> List[Instruction]:
        """Get the evolution layers given the list of operators and parameters.

        Since computing the evolution layers can be parallelized, this calls a parallel map
        with the function ``_get_single_evolution_layer``.

        Args:
            operators: The list of operators for the evolution instructions.
            params: The evolution times.

        Returns:
            A list containing all evolution instructions.
        """

        operator_param_pairs = list(zip(operators, params))
        all_evolution_instructions = parallel_map(task=self._get_single_evolution_layer,
                                                  values=operator_param_pairs,
                                                  num_processes=aqua_globals.num_processes)
        return all_evolution_instructions
