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
from qiskit.chemistry.core import MolecularGroundStateResult
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.chemistry.qubit_transformations import QubitOperatorTransformation

from .mes_factory import MESFactory


class MinimumEigensolverGroundStateCalculation(GroundStateCalculation):
    """TODO"""

    def __init__(self, transformation: QubitOperatorTransformation,
                 solver: Union[MinimumEigensolver, MESFactory]) -> None:
        """
        Args:
            transformation: TODO
            solver: TODO
        """
        super().__init__(transformation)
        self._solver = solver

    @property
    def solver(self) -> Union[MinimumEigensolver, MESFactory]:
        """Get the minimum eigensolver or factory."""
        return self._solver

    def returns_groundstate(self) -> bool:
        """TODO"""
        return False

    def compute_ground_state(self, driver: BaseDriver) -> MolecularGroundStateResult:
        # TODO MolecularGroundStateResult should become generic for bosonic and fermionic
        # Hamiltonains
        """Compute Ground State properties.

        Args:
            driver: A chemistry driver.

        Returns:
            A molecular ground state result
        """
        operator, aux_operators = self.transformation.transform(driver)

        if isinstance(self._solver, MESFactory):
            # this must be called after transformation.transform
            solver = self._solver.get_solver(self.transformation)  # TODO and driver?
        else:
            solver = self._solver

        # TODO shouldn't this rather raise a warning?
        aux_operators = aux_operators if solver.supports_aux_operators() else None

        raw_gs_result = solver.compute_minimum_eigenvalue(operator, aux_operators)

        # TODO WOR: where should this post processing be coming from?
        # The post processing is now in the tranformation so that it is fermionic or bosonic
        # gsc_result = self._transformation.interpret(raw_gs_result['energy'], r['aux_values'],
        # groundstate)  # gs = array/circuit+params
        return self.transformation.interpret(raw_gs_result)
        # (energy, aux_values, groundsntate)
