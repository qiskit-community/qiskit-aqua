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

"""The electronic structure result."""

from typing import List, Optional, Tuple, cast, Union

import logging
import numpy as np

from qiskit.chemistry import QMolecule

from .eigenstate_result import EigenstateResult

logger = logging.getLogger(__name__)

# A dipole moment, when present as X, Y and Z components will normally have float values for all
# the components. However when using Z2Symmetries, if the dipole component operator does not
# commute with the symmetry then no evaluation is done and None will be used as the 'value'
# indicating no measurement of the observable took place
DipoleTuple = Tuple[Optional[float], Optional[float], Optional[float]]


class ElectronicStructureResult(EigenstateResult):
    """The electronic structure result."""

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

    # TODO we need to be able to extract the statevector or the optimal parameters that can
    # construct the circuit of the GS from here (if the algorithm supports this)

    @property
    def total_energies(self) -> np.ndarray:
        """ Returns ground state energy if nuclear_repulsion_energy is available from driver """
        nre = self.nuclear_repulsion_energy if self.nuclear_repulsion_energy is not None else 0
        # Adding float to np.ndarray adds it to each entry
        return self.electronic_energies + nre

    @property
    def electronic_energies(self) -> np.ndarray:
        """ Returns electronic part of ground state energy """
        # TODO the fact that this property is computed on the fly breaks the `.combine()`
        # functionality
        # Adding float to np.ndarray adds it to each entry
        return (self.computed_energies
                + self.ph_extracted_energy
                + self.frozen_extracted_energy)

    @property
    def computed_energies(self) -> np.ndarray:
        """ Returns computed electronic part of ground state energy """
        return self.get('computed_energies')

    @computed_energies.setter
    def computed_energies(self, value: np.ndarray) -> None:
        """ Sets computed electronic part of ground state energy """
        self.data['computed_energies'] = value

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
    def total_dipole_moment(self) -> Optional[List[float]]:
        """ Returns total dipole of moment """
        if self.dipole_moment is None:
            return None  # No dipole at all
        tdm: List[float] = []
        for dip in self.dipole_moment:
            if np.any(np.equal(list(dip), None)):
                tdm.append(None)  # One or more components in the dipole is None
            else:
                tdm.append(np.sqrt(np.sum(np.power(list(dip), 2))))
        return tdm

    @property
    def total_dipole_moment_in_debye(self) -> Optional[List[float]]:
        """ Returns total dipole of moment in Debye """
        tdm = self.total_dipole_moment
        if tdm is None:
            return None
        return [dip / QMolecule.DEBYE for dip in tdm]

    @property
    def dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """ Returns dipole moment """
        edm = self.electronic_dipole_moment
        if self.reverse_dipole_sign:
            edm = [cast(DipoleTuple, tuple(-1 * x if x is not None else None for x in dip))
                   for dip in edm]
        return [_dipole_tuple_add(dip, self.nuclear_dipole_moment) for dip in edm]

    @property
    def dipole_moment_in_debye(self) -> Optional[List[DipoleTuple]]:
        """ Returns dipole moment in Debye """
        dipm = self.dipole_moment
        if dipm is None:
            return None
        dipmd = []
        for dip in dipm:
            dipmd0 = dip[0]/QMolecule.DEBYE if dip[0] is not None else None
            dipmd1 = dip[1]/QMolecule.DEBYE if dip[1] is not None else None
            dipmd2 = dip[2]/QMolecule.DEBYE if dip[2] is not None else None
            dipmd += [(dipmd0, dipmd1, dipmd2)]
        return dipmd

    @property
    def electronic_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """ Returns electronic dipole moment """
        return [_dipole_tuple_add(comp_dip, _dipole_tuple_add(ph_dip, frozen_dip)) for
                comp_dip, ph_dip, frozen_dip in zip(self.computed_dipole_moment,
                                                    self.ph_extracted_dipole_moment,
                                                    self.frozen_extracted_dipole_moment)]

    @property
    def computed_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """ Returns computed electronic part of dipole moment """
        return self.get('computed_dipole_moment')

    @computed_dipole_moment.setter
    def computed_dipole_moment(self, value: List[DipoleTuple]) -> None:
        """ Sets computed electronic part of dipole moment """
        self.data['computed_dipole_moment'] = value

    @property
    def ph_extracted_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """ Returns particle hole extracted part of dipole moment """
        return self.get('ph_extracted_dipole_moment')

    @ph_extracted_dipole_moment.setter
    def ph_extracted_dipole_moment(self, value: List[DipoleTuple]) -> None:
        """ Sets particle hole extracted part of dipole moment """
        self.data['ph_extracted_dipole_moment'] = value

    @property
    def frozen_extracted_dipole_moment(self) -> Optional[List[DipoleTuple]]:
        """ Returns frozen extracted part of dipole moment """
        return self.get('frozen_extracted_dipole_moment')

    @frozen_extracted_dipole_moment.setter
    def frozen_extracted_dipole_moment(self, value: List[DipoleTuple]) -> None:
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
    def total_angular_momentum(self) -> Optional[List[float]]:
        """ Returns total angular momentum (S^2) """
        return self.get('total_angular_momentum')

    @total_angular_momentum.setter
    def total_angular_momentum(self, value: List[float]) -> None:
        """ Sets total angular momentum """
        self.data['total_angular_momentum'] = value

    @property
    def spin(self) -> Optional[List[float]]:
        """ Returns computed spin """
        if self.total_angular_momentum is None:
            return None
        spin = []
        for total_angular_momentum in self.total_angular_momentum:
            spin.append((-1.0 + np.sqrt(1 + 4 * total_angular_momentum)) / 2)
        return spin

    @property
    def num_particles(self) -> Optional[List[float]]:
        """ Returns measured number of particles """
        return self.get('num_particles')

    @num_particles.setter
    def num_particles(self, value: List[float]) -> None:
        """ Sets measured number of particles """
        self.data['num_particles'] = value

    @property
    def magnetization(self) -> Optional[List[float]]:
        """ Returns measured magnetization """
        return self.get('magnetization')

    @magnetization.setter
    def magnetization(self, value: List[float]) -> None:
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
                     format(round(self.electronic_energies[0], 12)))
        lines.append('  - computed part:      {}'.
                     format(round(self.computed_energies[0], 12)))
        lines.append('  - frozen energy part: {}'.
                     format(round(self.frozen_extracted_energy, 12)))
        lines.append('  - particle hole part: {}'
                     .format(round(self.ph_extracted_energy, 12)))
        if self.nuclear_repulsion_energy is not None:
            lines.append('~ Nuclear repulsion energy (Hartree): {}'.
                         format(round(self.nuclear_repulsion_energy, 12)))
            lines.append('> Total ground state energy (Hartree): {}'.
                         format(round(self.total_energies[0], 12)))

        if len(self.computed_energies) > 1:
            lines.append(' ')
            lines.append('=== EXCITED STATE ENERGIES ===')
            lines.append(' ')
            for idx, (elec_energy, total_energy) in enumerate(zip(self.electronic_energies[1:],
                                                                  self.total_energies[1:])):
                lines.append('{: 3d}: '.format(idx+1))
                lines.append('* Electronic excited state energy (Hartree): {}'.
                             format(round(elec_energy, 12)))
                lines.append('> Total excited state energy (Hartree): {}'.
                             format(round(total_energy, 12)))

        if self.has_observables():
            lines.append(' ')
            lines.append('=== MEASURED OBSERVABLES ===')
            lines.append(' ')
            for idx, (num_particles, spin, total_angular_momentum, magnetization) in enumerate(zip(
                    self.num_particles, self.spin, self.total_angular_momentum,
                    self.magnetization)):
                line = '{: 3d}: '.format(idx)
                if num_particles is not None:
                    line += ' # Particles: {:.3f}'.format(num_particles)
                if spin is not None:
                    line += ' S: {:.3f}'.format(spin)
                if total_angular_momentum is not None:
                    line += ' S^2: {:.3f}'.format(total_angular_momentum)
                if magnetization is not None:
                    line += ' M: {:.3f}'.format(magnetization)
                lines.append(line)

        if self.has_dipole():
            lines.append(' ')
            lines.append('=== DIPOLE MOMENTS ===')
            lines.append(' ')
            if self.nuclear_dipole_moment is not None:
                lines.append('~ Nuclear dipole moment (a.u.): {}'
                             .format(_dipole_to_string(self.nuclear_dipole_moment)))
                lines.append(' ')
            for idx, (elec_dip, comp_dip, frozen_dip, ph_dip, dip, tot_dip, dip_db, tot_dip_db) in \
                    enumerate(zip(
                            self.electronic_dipole_moment, self.computed_dipole_moment,
                            self.frozen_extracted_dipole_moment, self.ph_extracted_dipole_moment,
                            self.dipole_moment, self.total_dipole_moment,
                            self.dipole_moment_in_debye, self.total_dipole_moment_in_debye)):
                lines.append('{: 3d}: '.format(idx))
                lines.append('  * Electronic dipole moment (a.u.): {}'
                             .format(_dipole_to_string(elec_dip)))
                lines.append('    - computed part:      {}'
                             .format(_dipole_to_string(comp_dip)))
                lines.append('    - frozen energy part: {}'
                             .format(_dipole_to_string(frozen_dip)))
                lines.append('    - particle hole part: {}'
                             .format(_dipole_to_string(ph_dip)))
                if self.nuclear_dipole_moment is not None:
                    lines.append('  > Dipole moment (a.u.): {}  Total: {}'
                                 .format(_dipole_to_string(dip), _float_to_string(tot_dip)))
                    lines.append('                 (debye): {}  Total: {}'
                                 .format(_dipole_to_string(dip_db), _float_to_string(tot_dip_db)))
                lines.append(' ')

        return lines


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
