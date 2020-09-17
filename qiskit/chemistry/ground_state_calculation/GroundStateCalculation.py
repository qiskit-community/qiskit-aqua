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


class GroundStateCalculation(ABC):
    """The ground state calculation interface."""

    def __init__(self,
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None)->None:
        """
        Args:
            transformation: full or particle_hole
            qubit_mapping: jordan_wigner, parity or bravyi_kitaev
            two_qubit_reduction: Whether two qubit reduction should be used,
                                  when parity mapping only
            freeze_core: Whether to freeze core orbitals when possible
            orbital_reduction: Orbital list to be frozen or removed
            z2symmetry_reduction: If z2 symmetry reduction should be applied to the qubit operators
                 that are computed. Setting 'auto' will
                 use an automatic computation of the correct sector. If from other experiments, with
                 the z2symmetry logic, the sector is known, then the tapering values of that sector
                 can be provided (a list of int of values -1, and 1). The default is None
                 meaning no symmetry reduction is done.
                 See also :class:`~qiskit.chemistry.core.Hamiltonian` which has the core
                 processing behind this class.
        """
        self._transformation = transformation
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction
        self._z2symmetry_reduction = z2symmetry_reduction

        # this is to provide access to the internal Hamiltonian object in derived classes
        self._core = None

    def _transform(self, driver: BaseDriver) -> Tuple[LegacyBaseOperator, List]:
        """Constructs a qubit operator for a given chemical problem.

        Args:
            driver: A chemical driver instance encoding the molecular problem.

        Returns:
            The qubit operator and auxiliary operator list transformed based on the specified
            mapping.
        """
        q_molecule = driver.run()

        self._core = Hamiltonian(transformation=self._transformation,
                                 qubit_mapping=self._qubit_mapping,
                                 two_qubit_reduction=self._two_qubit_reduction,
                                 freeze_core=self._freeze_core,
                                 orbital_reduction=self._orbital_reduction,
                                 z2symmetry_reduction=self._z2symmetry_reduction)

        operator, aux_operators = self._core.run(q_molecule)

        return operator, aux_operators

    @property
    def molecule_info(self) -> Dict:
        """Returns the molecular info stored in the core Hamiltonian."""
        return self._core.molecule_info

    @abstractmethod
    def compute_ground_state(self,
                             driver: BaseDriver,
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
