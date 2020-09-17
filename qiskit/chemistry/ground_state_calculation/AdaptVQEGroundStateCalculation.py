# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A ground state calculation employing the VQEAdapt algorithm.
"""

import logging
from typing import List, Optional, Callable, Union

from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.algorithms import VQEAdapt
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import (TransformationType, QubitMappingType, ChemistryOperator,
                                   MolecularGroundStateResult)
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.ground_state_calculation import GroundStateCalculation
from qiskit.providers import BaseBackend

logger = logging.getLogger(__name__)


class AdaptVQEGroundStateCalculation(GroundStateCalculation):
    """A ground state calculation employing the VQEAdapt algorithm."""
    def __init__(self,
                 quantum_instance: Union[QuantumInstance, BaseBackend],
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:
        """
        Args:
            quantum_instance: a quantum instance
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

        super().__init__(transformation, qubit_mapping, two_qubit_reduction, freeze_core,
                         orbital_reduction, z2symmetry_reduction)

        self._quantum_instance = quantum_instance
        # the solver object is used internally in order to be consistent with the
        # GroundStateCalculation implementation
        self._solver = None

    def compute_ground_state(self,
                             driver: BaseDriver,
                             callback: Optional[Callable[[List, int, str, bool, Z2Symmetries],
                                                         MinimumEigensolver]] = None
                             ) -> MolecularGroundStateResult:
        """Compute the ground state energy of the molecule that was supplied via the driver.

        Args:
            driver: A chemistry driver.
            callback: This argument will be ignored and is only provided for compatibility reasons!

        Returns:
            A molecular ground state result.
        """
        operator, aux_operators = self._transform(driver)

        if callback is not None:
            logger.warning("The `callback` option is only provided for compatibility reasons and \
                           has no effect in this context!")

        # gather required data from molecule info
        num_particles = self.molecule_info[ChemistryOperator.INFO_NUM_PARTICLES]
        num_orbitals = self.molecule_info[ChemistryOperator.INFO_NUM_ORBITALS]
        z2_symmetries = self.molecule_info[ChemistryOperator.INFO_Z2SYMMETRIES]

        # contract variational form base
        initial_state = HartreeFock(num_orbitals, num_particles, self._qubit_mapping.value,
                                    self._two_qubit_reduction, z2_symmetries.sq_list)
        var_form_base = UCCSD(num_orbitals=num_orbitals,
                              num_particles=num_particles,
                              initial_state=initial_state,
                              qubit_mapping=self._qubit_mapping.value,
                              two_qubit_reduction=self._two_qubit_reduction,
                              z2_symmetries=z2_symmetries)

        # initialize the adaptive VQE algorithm with the specified quantum instance
        self._solver = VQEAdapt(var_form_base=var_form_base)
        self._solver.quantum_instance = self._quantum_instance

        # run the algorithm and post-process the result
        raw_gs_result = self._solver.compute_minimum_eigenvalue(operator, aux_operators)
        return self._core.process_algorithm_result(raw_gs_result)
