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

from typing import Union

from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation
from qiskit.chemistry.results import EigenstateResult

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

    def compute_groundstate(self, driver: BaseDriver) -> EigenstateResult:
        """Compute Ground State properties.

        Args:
            driver: a chemistry driver object which defines the chemical problem that is to be
                    solved by this calculation.

        Returns:
            An eigenstate result. Depending on the transformation this can be an electronic
            structure or bosonic result.
        """
        operator, aux_operators = self.transformation.transform(driver)

        if isinstance(self._solver, MESFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(self.transformation)
        else:
            solver = self._solver

        aux_operators = aux_operators if solver.supports_aux_operators() else None

        raw_mes_result = solver.compute_minimum_eigenvalue(operator, aux_operators)

        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_mes_result
        eigenstate_result.eigenvalue = raw_mes_result.eigenvalue
        eigenstate_result.aux_values = raw_mes_result.aux_operator_eigenvalues
        result = self.transformation.interpret(eigenstate_result)
        return result
