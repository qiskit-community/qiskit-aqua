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

"""The calculation of excited states via an Eigensolver algorithm"""

import logging
from typing import List, Union, Optional, Any

from qiskit.aqua.algorithms import Eigensolver
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.results import (EigenstateResult,
                                      ElectronicStructureResult,
                                      VibronicStructureResult)
from qiskit.chemistry.transformations import Transformation

from .excited_states_solver import ExcitedStatesSolver
from .eigensolver_factories import EigensolverFactory

logger = logging.getLogger(__name__)


class ExcitedStatesEigensolver(ExcitedStatesSolver):
    """The calculation of excited states via an Eigensolver algorithm"""

    def __init__(self, transformation: Transformation,
                 solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """

        Args:
            transformation: Qubit Operator Transformation
            solver: Minimum Eigensolver or MESFactory object.
        """
        self._transformation = transformation
        self._solver = solver

    @property
    def solver(self) -> Union[Eigensolver, EigensolverFactory]:
        """Returns the minimum eigensolver or factory."""
        return self._solver

    @solver.setter
    def solver(self, solver: Union[Eigensolver, EigensolverFactory]) -> None:
        """Sets the minimum eigensolver or factory."""
        self._solver = solver

    @property
    def transformation(self) -> Transformation:
        """Returns the transformation used to obtain a qubit operator from the molecule."""
        return self._transformation

    @transformation.setter
    def transformation(self, transformation: Transformation) -> None:
        """Sets the transformation used to obtain a qubit operator from the molecule."""
        self._transformation = transformation

    def solve(self, driver: BaseDriver,
              aux_operators: Optional[List[Any]] = None
              ) -> Union[ElectronicStructureResult, VibronicStructureResult]:
        """Compute Ground and Excited States properties.

        Args:
            driver: a chemistry driver object which defines the chemical problem that is to be
                    solved by this calculation.
            aux_operators: Additional auxiliary operators to evaluate. Must be of type
                ``FermionicOperator`` if the qubit transformation is fermionic and of type
                ``BosonicOperator`` it is bosonic.

        Raises:
            NotImplementedError: If an operator in ``aux_operators`` is not of type
                ``FermionicOperator``.

        Returns:
            An eigenstate result. Depending on the transformation this can be an electronic
            structure or bosonic result.
        """
        if aux_operators is not None:
            if any(not isinstance(op, (WeightedPauliOperator, FermionicOperator))
                   for op in aux_operators):
                raise NotImplementedError('Currently only fermionic problems are supported.')

        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_operators`` contains not only the transformed ``aux_operators`` passed
        # by the user but also additional ones from the transformation
        operator, aux_operators = self.transformation.transform(driver, aux_operators)

        if isinstance(self._solver, EigensolverFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(self.transformation)
        else:
            solver = self._solver

        # if the eigensolver does not support auxiliary operators, reset them
        if not solver.supports_aux_operators():
            aux_operators = None

        raw_es_result = solver.compute_eigenvalues(operator, aux_operators)

        eigenstate_result = EigenstateResult()
        eigenstate_result.raw_result = raw_es_result
        eigenstate_result.eigenenergies = raw_es_result.eigenvalues
        eigenstate_result.eigenstates = raw_es_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = raw_es_result.aux_operator_eigenvalues
        result = self.transformation.interpret(eigenstate_result)
        return result
