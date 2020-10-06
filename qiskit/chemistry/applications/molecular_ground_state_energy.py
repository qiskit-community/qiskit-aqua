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

""" Molecular ground state energy  chemistry application """

from typing import List, Optional, Callable, Union

from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import MinimumEigensolver, VQE
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator, MolecularGroundStateResult)
from qiskit.chemistry.drivers import BaseDriver


class MolecularGroundStateEnergy:
    """ Molecular ground state energy chemistry application """

    def __init__(self,
                 driver: BaseDriver,
                 solver: Optional[MinimumEigensolver] = None,
                 transformation: TransformationType = TransformationType.FULL,
                 qubit_mapping: QubitMappingType = QubitMappingType.PARITY,
                 two_qubit_reduction: bool = True,
                 freeze_core: bool = False,
                 orbital_reduction: Optional[List[int]] = None,
                 z2symmetry_reduction: Optional[Union[str, List[int]]] = None) -> None:
        """
        Args:
            driver: Chemistry driver
            solver: An Aqua MinimumEigensolver. This can be provided on the constructor or
                    via the solver property, or via the callback on :meth:`compute_energy`
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
        self._driver = driver
        self._solver = solver
        self._transformation = transformation
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction
        self._z2symmetry_reduction = z2symmetry_reduction

    @property
    def driver(self) -> BaseDriver:
        """ Returns chemistry driver """
        return self._driver

    @driver.setter
    def driver(self, driver: BaseDriver) -> None:
        self._driver = driver

    @property
    def solver(self) -> MinimumEigensolver:
        """ Returns minimum eigen solver """
        return self._solver

    @solver.setter
    def solver(self, solver: MinimumEigensolver) -> None:
        """ Sets minimum eigen solver """
        self._solver = solver

    def compute_energy(self,
                       callback: Optional[Callable[[List, int, str, bool, Z2Symmetries],
                                                   MinimumEigensolver]] = None
                       ) -> MolecularGroundStateResult:
        """
        Compute the ground state energy of the molecule that was supplied via the driver

        Args:
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
        if self.solver is None and callback is None:
            raise QiskitChemistryError('MinimumEigensolver was not provided')

        q_molecule = self.driver.run()
        core = Hamiltonian(transformation=self._transformation,
                           qubit_mapping=self._qubit_mapping,
                           two_qubit_reduction=self._two_qubit_reduction,
                           freeze_core=self._freeze_core,
                           orbital_reduction=self._orbital_reduction,
                           z2symmetry_reduction=self._z2symmetry_reduction)
        operator, aux_operators = core.run(q_molecule)

        if callback is not None:
            num_particles = core.molecule_info[ChemistryOperator.INFO_NUM_PARTICLES]
            num_orbitals = core.molecule_info[ChemistryOperator.INFO_NUM_ORBITALS]
            two_qubit_reduction = core.molecule_info[ChemistryOperator.INFO_TWO_QUBIT_REDUCTION]
            z2_symmetries = core.molecule_info[ChemistryOperator.INFO_Z2SYMMETRIES]
            self.solver = callback(num_particles, num_orbitals,
                                   self._qubit_mapping.value, two_qubit_reduction,
                                   z2_symmetries)

        aux_operators = aux_operators if self.solver.supports_aux_operators() else None

        raw_result = self.solver.compute_minimum_eigenvalue(operator, aux_operators)
        return core.process_algorithm_result(raw_result)

    @staticmethod
    def get_default_solver(quantum_instance: Union[QuantumInstance, Backend, BaseBackend]) ->\
            Optional[Callable[[List, int, str, bool, Z2Symmetries], MinimumEigensolver]]:
        """
        Get the default solver callback that can be used with :meth:`compute_energy`
        Args:
            quantum_instance: A Backend/Quantum Instance for the solver to run on

        Returns:
            Default solver callback
        """
        def cb_default_solver(num_particles, num_orbitals,
                              qubit_mapping, two_qubit_reduction, z2_symmetries):
            """ Default solver """
            initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                        two_qubit_reduction, z2_symmetries.sq_list)
            var_form = UCCSD(num_orbitals=num_orbitals,
                             num_particles=num_particles,
                             initial_state=initial_state,
                             qubit_mapping=qubit_mapping,
                             two_qubit_reduction=two_qubit_reduction,
                             z2_symmetries=z2_symmetries)
            vqe = VQE(var_form=var_form)
            vqe.quantum_instance = quantum_instance
            return vqe
        return cb_default_solver
