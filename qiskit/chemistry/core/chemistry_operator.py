# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module contains the definition of a base class for a chemistry operator.
Such an operator takes a QMolecule and produces an input for
a quantum algorithm
"""
from abc import ABC, abstractmethod
import logging
from typing import Union, List, Tuple, Optional
import numpy as np

from qiskit.aqua.algorithms import MinimumEigensolverResult, EigensolverResult, AlgorithmResult
from qiskit.chemistry import QMolecule

logger = logging.getLogger(__name__)

# A dipole moment, when present as X, Y and Z components will normally have float values for all
# the components. However when using Z2Symmetries, if the dipole component operator does not
# commute with the symmetry then no evaluation is done and None will be used as the 'value'
# indicating no measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class ChemistryOperator(ABC):
    """
    Base class for ChemistryOperator.
    """

    INFO_NUM_PARTICLES = 'num_particles'
    INFO_NUM_ORBITALS = 'num_orbitals'
    INFO_TWO_QUBIT_REDUCTION = 'two_qubit_reduction'
    INFO_Z2SYMMETRIES = 'z2symmetries'

    @abstractmethod
    def __init__(self):
        self._molecule_info = {}

    @abstractmethod
    def run(self, qmolecule):
        """
        Convert the qmolecule, according to the ChemistryOperator, into an Operator
        that can be given to a QuantumAlgorithm

        Args:
            qmolecule (QMolecule): from a chemistry driver

        Returns:
            Tuple: (qubit_op, aux_ops)
        """
        raise NotImplementedError

    def process_algorithm_result(
            self, algo_result: Union[dict,
                                     MinimumEigensolverResult,
                                     EigensolverResult]) -> Union[Tuple[List[str], dict],
                                                                  'MolecularGroundStateResult',
                                                                  'MolecularExcitedStatesResult']:
        """
        Takes the algorithm result and processes it as required, e.g. by
        combination of any parts that were classically computed, for the
        final result.

        Args:
            algo_result: Result from algorithm

        Returns:
            Final chemistry result computed from the algorithm result
        """
        if isinstance(algo_result, MinimumEigensolverResult):
            return self._process_algorithm_result(algo_result)
        elif isinstance(algo_result, EigensolverResult):
            return self._process_algorithm_result(algo_result)
        else:
            lines, result = self._process_algorithm_result(algo_result)
            result['algorithm_retvals'] = algo_result
            return lines, result

    @abstractmethod
    def _process_algorithm_result(self, algo_result):
        raise NotImplementedError

    @property
    def molecule_info(self):
        """ returns molecule info """
        return self._molecule_info

    def _add_molecule_info(self, key, value):
        self._molecule_info[key] = value


class MolecularChemistryResult(AlgorithmResult):
    """
    Molecular chemistry Result

    Energies are in Hartree and dipole moments in A.U unless otherwise stated.
    """

    @property
    def algorithm_result(self) -> AlgorithmResult:
        """ Returns raw algorithm result """
        return self.get('algorithm_result')

    @algorithm_result.setter
    def algorithm_result(self, value: AlgorithmResult) -> None:
        """ Sets raw algorithm result """
        self.data['algorithm_result'] = value

    @property
    def hartree_fock_energy(self) -> float:
        """ Returns Hartree-Fock energy """
        return self.get('hartree_fock_energy')

    @hartree_fock_energy.setter
    def hartree_fock_energy(self, value: float) -> None:
        """ Sets Hartree-Fock energy """
        self.data['hartree_fock_energy'] = value

    @property
    def nuclear_repulsion_energy(self) -> Optional[float]:
        """ Returns nuclear repulsion energy when available from driver """
        return self.get('nuclear_repulsion_energy')

    @nuclear_repulsion_energy.setter
    def nuclear_repulsion_energy(self, value: float) -> None:
        """ Sets nuclear repulsion energy """
        self.data['nuclear_repulsion_energy'] = value

    @property
    def nuclear_dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns nuclear dipole moment X,Y,Z components in A.U when available from driver """
        return self.get('nuclear_dipole_moment')

    @nuclear_dipole_moment.setter
    def nuclear_dipole_moment(self, value: DipoleTuple) -> None:
        """ Sets nuclear dipole moment in A.U """
        self.data['nuclear_dipole_moment'] = value


class MolecularGroundStateResult(MolecularChemistryResult):
    """
    Molecular Ground State Energy Result.

    Energies are in Hartree and dipole moments in A.U unless otherwise stated.
    """

    @property
    def energy(self) -> Optional[float]:
        """ Returns ground state energy if nuclear_repulsion_energy is available from driver """
        nre = self.nuclear_repulsion_energy
        return self.electronic_energy + nre if nre is not None else None

    @property
    def electronic_energy(self) -> float:
        """ Returns electronic part of ground state energy """
        return (self.computed_electronic_energy
                + self.ph_extracted_energy
                + self.frozen_extracted_energy)

    @property
    def computed_electronic_energy(self) -> float:
        """ Returns computed electronic part of ground state energy """
        return self.get('computed_electronic_energy')

    @computed_electronic_energy.setter
    def computed_electronic_energy(self, value: float) -> None:
        """ Sets computed electronic part of ground state energy """
        self.data['computed_electronic_energy'] = value

    @property
    def ph_extracted_energy(self) -> float:
        """ Returns particle hole extracted part of ground state energy """
        return self.get('ph_extracted_energy')

    @ph_extracted_energy.setter
    def ph_extracted_energy(self, value: float) -> None:
        """ Sets particle hole extracted part of ground state energy """
        self.data['ph_extracted_energy'] = value

    @property
    def frozen_extracted_energy(self) -> float:
        """ Returns frozen extracted part of ground state energy """
        return self.get('frozen_extracted_energy')

    @frozen_extracted_energy.setter
    def frozen_extracted_energy(self, value: float) -> None:
        """ Sets frozen extracted part of ground state energy """
        self.data['frozen_extracted_energy'] = value

    # Dipole moment results. Note dipole moments of tuples of X, Y and Z components. Chemistry
    # drivers either support dipole integrals or not. Note that when using Z2 symmetries of

    def has_dipole(self) -> bool:
        """ Returns whether dipole moment is present in result or not """
        return self.nuclear_dipole_moment is not None and self.electronic_dipole_moment is not None

    @property
    def reverse_dipole_sign(self) -> bool:
        """ Returns if electronic dipole moment sign should be reversed when adding to nuclear """
        return self.get('reverse_dipole_sign')

    @reverse_dipole_sign.setter
    def reverse_dipole_sign(self, value: bool) -> None:
        """ Sets if electronic dipole moment sign should be reversed when adding to nuclear """
        self.data['reverse_dipole_sign'] = value

    @property
    def total_dipole_moment(self) -> Optional[float]:
        """ Returns total dipole of moment """
        if self.dipole_moment is None:
            return None  # No dipole at all
        if np.any(np.equal(list(self.dipole_moment), None)):
            return None  # One or more components in the dipole is None
        return np.sqrt(np.sum(np.power(list(self.dipole_moment), 2)))

    @property
    def total_dipole_moment_in_debye(self) -> Optional[float]:
        """ Returns total dipole of moment in Debye """
        tdm = self.total_dipole_moment
        return tdm / QMolecule.DEBYE if tdm is not None else None

    @property
    def dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns dipole moment """
        edm = self.electronic_dipole_moment
        if self.reverse_dipole_sign:
            edm = tuple(-1 * x if x is not None else None for x in edm)
        return _dipole_tuple_add(edm, self.nuclear_dipole_moment)

    @property
    def dipole_moment_in_debye(self) -> Optional[DipoleTuple]:
        """ Returns dipole moment in Debye """
        dipm = self.dipole_moment
        if dipm is None:
            return None
        dipmd0 = dipm[0]/QMolecule.DEBYE if dipm[0] is not None else None
        dipmd1 = dipm[1]/QMolecule.DEBYE if dipm[1] is not None else None
        dipmd2 = dipm[2]/QMolecule.DEBYE if dipm[2] is not None else None
        return dipmd0, dipmd1, dipmd2

    @property
    def electronic_dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns electronic dipole moment """
        return _dipole_tuple_add(self.computed_dipole_moment,
                                 _dipole_tuple_add(self.ph_extracted_dipole_moment,
                                                   self.frozen_extracted_dipole_moment))

    @property
    def computed_dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns computed electronic part of dipole moment """
        return self.get('computed_dipole_moment')

    @computed_dipole_moment.setter
    def computed_dipole_moment(self, value: DipoleTuple) -> None:
        """ Sets computed electronic part of dipole moment """
        self.data['computed_dipole_moment'] = value

    @property
    def ph_extracted_dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns particle hole extracted part of dipole moment """
        return self.get('ph_extracted_dipole_moment')

    @ph_extracted_dipole_moment.setter
    def ph_extracted_dipole_moment(self, value: DipoleTuple) -> None:
        """ Sets particle hole extracted part of dipole moment """
        self.data['ph_extracted_dipole_moment'] = value

    @property
    def frozen_extracted_dipole_moment(self) -> Optional[DipoleTuple]:
        """ Returns frozen extracted part of dipole moment """
        return self.get('frozen_extracted_dipole_moment')

    @frozen_extracted_dipole_moment.setter
    def frozen_extracted_dipole_moment(self, value: DipoleTuple) -> None:
        """ Sets frozen extracted part of dipole moment """
        self.data['frozen_extracted_dipole_moment'] = value

    # Other measured operators. If these are not evaluated then None will be returned
    # instead of any measured value.

    def has_observables(self):
        """ Returns whether result has aux op observables such as spin, num particles """
        return self.total_angular_momentum is not None \
            or self.num_particles is not None \
            or self.magnetization is not None

    @property
    def total_angular_momentum(self) -> Optional[float]:
        """ Returns total angular momentum (S^2) """
        return self.get('total_angular_momentum')

    @total_angular_momentum.setter
    def total_angular_momentum(self, value: float) -> None:
        """ Sets total angular momentum """
        self.data['total_angular_momentum'] = value

    @property
    def spin(self) -> Optional[float]:
        """ Returns computed spin """
        if self.total_angular_momentum is None:
            return None
        return (-1.0 + np.sqrt(1 + 4 * self.total_angular_momentum)) / 2

    @property
    def num_particles(self) -> Optional[float]:
        """ Returns measured number of particles """
        return self.get('num_particles')

    @num_particles.setter
    def num_particles(self, value: float) -> None:
        """ Sets measured number of particles """
        self.data['num_particles'] = value

    @property
    def magnetization(self) -> Optional[float]:
        """ Returns measured magnetization """
        return self.get('magnetization')

    @magnetization.setter
    def magnetization(self, value: float) -> None:
        """ Sets measured magnetization """
        self.data['magnetization'] = value

    def __str__(self) -> str:
        """ Printable formatted result """
        return '\n'.join(self.formatted)

    @property
    def formatted(self) -> List[str]:
        """ Formatted result as a list of strings """
        lines = []
        lines.append('=== GROUND STATE ENERGY ===')
        lines.append(' ')
        lines.append('* Electronic ground state energy (Hartree): {}'.
                     format(round(self.electronic_energy, 12)))
        lines.append('  - computed part:      {}'.
                     format(round(self.computed_electronic_energy, 12)))
        lines.append('  - frozen energy part: {}'.
                     format(round(self.frozen_extracted_energy, 12)))
        lines.append('  - particle hole part: {}'
                     .format(round(self.ph_extracted_energy, 12)))
        if self.nuclear_repulsion_energy is not None:
            lines.append('~ Nuclear repulsion energy (Hartree): {}'.
                         format(round(self.nuclear_repulsion_energy, 12)))
            lines.append('> Total ground state energy (Hartree): {}'.
                         format(round(self.energy, 12)))
        if self.has_observables():
            line = '  Measured::'
            if self.num_particles is not None:
                line += ' # Particles: {:.3f}'.format(self.num_particles)
            if self.spin is not None:
                line += ' S: {:.3f}'.format(self.spin)
            if self.total_angular_momentum is not None:
                line += ' S^2: {:.3f}'.format(self.total_angular_momentum)
            if self.magnetization is not None:
                line += ' M: {:.5f}'.format(self.magnetization)
            lines.append(line)

        if self.has_dipole():
            lines.append(' ')
            lines.append('=== DIPOLE MOMENT ===')
            lines.append(' ')
            lines.append('* Electronic dipole moment (a.u.): {}'
                         .format(_dipole_to_string(self.electronic_dipole_moment)))
            lines.append('  - computed part:      {}'
                         .format(_dipole_to_string(self.computed_dipole_moment)))
            lines.append('  - frozen energy part: {}'
                         .format(_dipole_to_string(self.frozen_extracted_dipole_moment)))
            lines.append('  - particle hole part: {}'
                         .format(_dipole_to_string(self.ph_extracted_dipole_moment)))
            if self.nuclear_dipole_moment is not None:
                lines.append('~ Nuclear dipole moment (a.u.): {}'
                             .format(_dipole_to_string(self.nuclear_dipole_moment)))
                lines.append('> Dipole moment (a.u.): {}  Total: {}'
                             .format(_dipole_to_string(self.dipole_moment),
                                     _float_to_string(self.total_dipole_moment)))
                lines.append('               (debye): {}  Total: {}'
                             .format(_dipole_to_string(self.dipole_moment_in_debye),
                                     _float_to_string(self.total_dipole_moment_in_debye)))
        return lines


class MolecularExcitedStatesResult(MolecularChemistryResult):
    """
    Molecular Excited States Result

    Energies are in Hartree and dipole moments in A.U unless otherwise stated.
    """
    # TODO This needs completing once EigenSolver interface/result is final
    @property
    def energies(self) -> Tuple:
        """ Returns ground state energy """
        return self.get('energies')

    @energies.setter
    def energies(self, value: Tuple) -> None:
        """ Sets ground state energy """
        self.data['energies'] = value


def _dipole_tuple_add(x: Optional[DipoleTuple],
                      y: Optional[DipoleTuple]) -> Optional[DipoleTuple]:
    """ Utility to add two dipole tuples element-wise for dipole additions """
    if x is None or y is None:
        return None
    return _element_add(x[0], y[0]), _element_add(x[1], y[1]), _element_add(x[2], y[2])


def _element_add(x: Optional[float], y: Optional[float]):
    """ Add dipole elements where a value may be None then None is returned """
    return x + y if x is not None and y is not None else None


def _dipole_to_string(dipole: DipoleTuple):
    dips = [round(x, 8) if x is not None else x for x in dipole]
    value = '['
    for i, _ in enumerate(dips):
        value += _float_to_string(dips[i]) if dips[i] is not None else 'None'
        value += '  ' if i < len(dips)-1 else ']'
    return value


def _float_to_string(value: Optional[float], precision: int = 8) -> str:
    if value is None:
        return 'None'
    else:
        return '0.0' if value == 0 else ('{:.' + str(precision) + 'f}').format(value).rstrip('0')
