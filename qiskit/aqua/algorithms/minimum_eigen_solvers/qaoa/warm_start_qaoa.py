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

""" The Warm Start Quantum Approximate Optimization Algorithm. """
from typing import Optional, Union, Callable, List

import numpy as np
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend

from qiskit import QuantumCircuit
from qiskit.aqua import AquaError, QuantumInstance
from qiskit.aqua.algorithms import QAOA, VQE
from qiskit.aqua.algorithms.minimum_eigen_solvers.qaoa.var_form import QAOAVarForm
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.operators import CircuitOp, OperatorBase, LegacyBaseOperator, GradientBase, \
    ExpectationBase
from qiskit.aqua.utils.validation import validate_min


class WarmStartQAOA(VQE):

    def __init__(self,
                 operator: Union[OperatorBase, LegacyBaseOperator] = None,
                 optimizer: Optimizer = None,
                 p: int = 1,
                 initial_point: Optional[np.ndarray] = None,
                 gradient: Optional[Union[GradientBase, Callable[[Union[np.ndarray, List]],
                                                                 List]]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 max_evals_grouped: int = 1,
                 aux_operators: Optional[List[Optional[Union[OperatorBase, LegacyBaseOperator]]]] =
                 None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None,
                 initial_variables: Optional[List[float]] = None,
                 epsilon: float = 0.0,
                 ) -> None:
        if epsilon < 0. or epsilon > 0.5:
            raise AquaError('Epsilon for warm-start QAOA needs to be between 0 and 0.5.')
        validate_min('p', p, 1)

        self._p = p
        self._epsilon = epsilon
        self._create_initial_variables(initial_variables)
        self._create_initial_state()
        self._create_mixer(0)       # todo: beta!
        super().__init__(operator,
                         None,
                         optimizer,
                         initial_point=initial_point,
                         gradient=gradient,
                         expectation=expectation,
                         include_custom=include_custom,
                         max_evals_grouped=max_evals_grouped,
                         callback=callback,
                         quantum_instance=quantum_instance,
                         aux_operators=aux_operators)

    @VQE.operator.setter  # type: ignore
    def operator(self, operator: Union[OperatorBase, LegacyBaseOperator]) -> None:
        """ Sets operator """
        # Need to wipe the var_form in case number of qubits differs from operator.
        self.var_form = None
        # Setting with VQE's operator property
        super(QAOA, self.__class__).operator.__set__(self, operator)  # type: ignore
        self.var_form = QAOAVarForm(self.operator,
                                    self._p,
                                    initial_state=self._initial_state,
                                    mixer_operator=self._mixer_operator)

    def _create_initial_variables(self, initial_variables: List[float]) -> None:
        """
        Set the starting variable values to warm start QAOA. This creates the initial
        state quantum circuit and the mixer operator for warm start QAOA.

        Args:
            initial_variables: a solution obtained for the relaxed problem.

        Raises:
            AquaError: if ``epsilon`` is not specified for the warm start QAOA.
        """
        if self._epsilon is None:
            raise AquaError('Epsilon must be specified for the warm start QAOA')

        self._initial_variables = []

        for variable in initial_variables:
            if variable < self._epsilon:
                self._initial_variables.append(self._epsilon)
            elif variable > 1. - self._epsilon:
                self._initial_variables.append(1.-self._epsilon)
            else:
                self._initial_variables.append(variable)

    def _create_initial_state(self) -> None:
        circuit = QuantumCircuit(len(self._initial_variables))

        for index, relaxed_value in enumerate(self._initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))
            circuit.ry(theta, index)

        self._initial_state = circuit

    def _create_mixer(self, beta: float) -> None:
        """
        Creates the evolved mixer circuit as Ry(theta)Rz(-2beta)Ry(-theta)
        """
        circ = QuantumCircuit(len(self._initial_variables))

        for index, relaxed_value in enumerate(self._initial_variables):
            theta = 2 * np.arcsin(np.sqrt(relaxed_value))

            circ.ry(theta, index)
            circ.rz(-2.0 * beta, index)
            circ.ry(-theta, index)

        self._mixer_operator = CircuitOp(circ)
