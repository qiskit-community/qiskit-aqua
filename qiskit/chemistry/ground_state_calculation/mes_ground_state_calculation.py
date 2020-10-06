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

"""Ground state computation using a minimum eigensolver."""

from typing import Union, Dict, Any, Optional

from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation
from qiskit.chemistry.results import FermionicGroundStateResult

from .mes_factories import MESFactory


class MinimumEigensolverGroundStateCalculation(GroundStateCalculation):
    """Ground state computation using a minimum eigensolver."""

    def __init__(self, transformation: QubitOperatorTransformation,
                 solver: Union[MinimumEigensolver, MESFactory]) -> None:
        """

        Args:
            transformation: Qubit Operator Transformation
            solver: Minimum Eigensolver or MESFactory object, e.g. the VQEUCCSDFactory.
        """
        super().__init__(transformation)
        self._solver = solver

    @property
    def solver(self) -> Union[MinimumEigensolver, MESFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    @solver.setter
    def solver(self, solver: Union[MinimumEigensolver, MESFactory]) -> None:
        """Sets the minimum eigensolver or factory."""
        self._solver = solver

    def returns_groundstate(self) -> bool:
        """TODO
        whether the eigensolver returns the ground state or only ground state energy."""

        return False

    def compute_groundstate(self, driver: BaseDriver,
                            additional_operators: Optional[Dict[str, Any]] = None
                            ) -> FermionicGroundStateResult:
        """Compute Ground State properties.

        Args:
            driver: A chemistry driver.
            additional_operators: Additional auxiliary ``FermionicOperator``s to evaluate at the
                ground state.

        Raises:
            ValueError: If an operator in ``additional_operators`` is not of type
                ``FermionicOperator``.

        Returns:
            Ground state result TODO
        """
        if any(not isinstance(op, FermionicOperator) for op in additional_operators.values()):
            raise ValueError('The additional operators must be of type FermionicOperator.')

        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_operators`` contains not only the transformed ``aux_operators`` passed
        # by the user but also additional ones from the transformation
        operator, aux_operators = self.transformation.transform(driver, additional_operators)

        if isinstance(self._solver, MESFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(self.transformation)
        else:
            solver = self._solver

        # convert aux_operators to a list for the minimum eigensolver
        mes_aux_ops = list(aux_operators.values()) if solver.supports_aux_operators() else None
        raw_mes_result = solver.compute_minimum_eigenvalue(operator, mes_aux_ops)

        # convert the aux_values back to a dictionary
        aux_values = dict(zip(aux_operators.keys(), raw_mes_result.aux_operator_eigenvalues))

        result = FermionicGroundStateResult()
        result.raw_result = raw_mes_result
        result.computed_electronic_energy = raw_mes_result.eigenvalue.real
        result.aux_values = aux_values
        self.transformation.add_context(result)
        return result
