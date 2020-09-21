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

"""
A ground state calculation interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Dict, Callable

from qiskit.aqua.operators import LegacyBaseOperator
from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   MolecularGroundStateResult)
from qiskit.chemistry.qubit_transformations.qubit_operator_transformation import \
    QubitOperatorTransformation


class GroundStateCalculation(ABC):
    """The ground state calculation interface."""

    def __init__(self, transformation: QubitOperatorTransformation) -> None:
        """
        Args:
            transformation: transformation from driver to qubit operator (and aux. operators)
        """
        self._transformation = transformation

    @abstractmethod
    def compute_ground_state(self,
                             driver: BaseDriver, additional_ops,
                             callback: Optional[Callable[[List, int, str, bool, Z2Symmetries],
                                                         MinimumEigensolver]] = None
                             ) -> MolecularGroundStateResult:
        """
        Compute the ground state energy of the molecule that was supplied via the driver.

        Args:
            driver: a chemical driver
            callback: If not None will be called with the following values
                num_particles, num_orbitals, qubit_mapping, two_qubit_reduction, z2_symmetries
                in that order. This information can then be used to setup chemistry
                specific component(s) that are needed by the chosen MinimumEigensolver.
                The MinimumEigensolver can then be built and returned from this callback
                for use as the solver here.

        Returns:
            A molecular ground state result
        Raises:
            QiskitChemistryError: If no MinimumEigensolver was given and no callback is being
                                  used that could supply one instead.
        """
        raise NotImplementedError()

    #@abstractmethod
    #def returns_groundstate() -> bool:
    #    # needs to return whether this calculation only returns groundstate energies or also groundstates
    #    raise NotImplementedError
