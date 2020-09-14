# -*- coding: utf-8 -*-

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


from abc import ABC, abstractmethod
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator, MolecularGroundStateResult)
from qiskit.aqua.operators import Z2Symmetries


# class GroundStateCalculationResult():
#     def __init__(self, results_parameters):
#         self._results_parameters = results_parameters
#         # param1
#         # param2
#         # param3
#     # add getters... cf. other results classes


class GroundStateCalculation(ABC):
    """GroundStateCalculation"""

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

    def _transform(self, driver):
        """

        Args:
            driver:

        Returns:

        """
        # takes driver, applies specified mapping, returns qubit operator

        q_molecule = self.driver.run()
        core = Hamiltonian(transformation=self._transformation,
                           qubit_mapping=self._qubit_mapping,
                           two_qubit_reduction=self._two_qubit_reduction,
                           freeze_core=self._freeze_core,
                           orbital_reduction=self._orbital_reduction,
                           z2symmetry_reduction=self._z2symmetry_reduction)
        operator, aux_operators = core.run(q_molecule)

        return operator, aux_operators

    @abstractmethod
    def compute_ground_state(driver) -> MolecularGroundStateResult:

        raise NotImplementedError()