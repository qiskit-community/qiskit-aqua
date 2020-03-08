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

""" Molecular ground state energy  chemistry application """

from typing import List, Optional, Callable

from qiskit.aqua.algorithms import MinimumEigensolver, AlgorithmResult, MinimumEigensolverResult
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.core import (Hamiltonian, TransformationType, QubitMappingType,
                                   ChemistryOperator)
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
                 orbital_reduction: Optional[List[int]] = None) -> None:
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
        """
        self._driver = driver
        self._solver = solver
        self._transformation = transformation
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._freeze_core = freeze_core
        self._orbital_reduction = orbital_reduction

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
                       callback: Optional[Callable[[List, int,
                                                    str, bool], MinimumEigensolver]] = None
                       ) -> 'MolecularGroundStateEnergyResult':
        """
        Compute the ground state energy of the molecule that was supplied via the driver

        Args:
            callback: In not None will be called with the following values
                      num_particles, num_orbitals, qubit_mapping, two_qubit_reduction
                      in that order. This information can then be used to setup chemistry
                      specific component(s) that are needed by the chosen MinimumEigensolver.
                      The MinimumEigensolver can then be built and returned from this callback
                      for use as the solver here.

        Returns:
            A MolecularGroundStateEnergyResult
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
                           orbital_reduction=self._orbital_reduction)
        operator, aux_operators = core.run(q_molecule)

        if callback is not None:
            num_particles = core.molecule_info[ChemistryOperator.INFO_NUM_PARTICLES]
            num_orbitals = core.molecule_info[ChemistryOperator.INFO_NUM_ORBITALS]
            self.solver = callback(num_particles, num_orbitals,
                                   self._qubit_mapping.value, self._two_qubit_reduction)

        aux_operators = aux_operators if self.solver.supports_aux_operators() else None

        raw_result = self.solver.compute_minimum_eigenvalue(operator, aux_operators)
        lines, core_result = core.process_algorithm_result(raw_result)

        mgse = MolecularGroundStateEnergyResult()
        mgse.energy = core_result['energy']
        mgse.printable = lines
        mgse.raw_result = raw_result

        return mgse


class MolecularGroundStateEnergyResult(AlgorithmResult):
    """ Ground state energy result."""

    @property
    def energy(self) -> float:
        """ Returns energy """
        return self.get('energy')

    @energy.setter
    def energy(self, value: float) -> None:
        """ Sets energy """
        self.data['energy'] = value

    @property
    def printable(self) -> List[str]:
        """ Returns printable """
        return self.get('printable')

    @printable.setter
    def printable(self, value: List[str]) -> None:
        """ Sets printable """
        self.data['printable'] = value

    def print_result(self) -> None:
        """ Prints the printable result """
        print(*self.printable, sep='\n')

    @property
    def raw_result(self) -> MinimumEigensolverResult:
        """ Returns raw result from MinimumEigensolver """
        return self.get('raw_result')

    @raw_result.setter
    def raw_result(self, value: MinimumEigensolverResult) -> None:
        """ Sets raw result from MinimumEigensolver  """
        self.data['raw_result'] = value
