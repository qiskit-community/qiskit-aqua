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

"""FCIDump Driver."""

from typing import List, Optional
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry import QiskitChemistryError, QMolecule
from .dumper import dump
from .parser import parse


class FCIDumpDriver(BaseDriver):
    """
    Qiskit chemistry driver reading an FCIDump file.

    The FCIDump format is partially defined in Knowles1989.

    References:
        Knowles1989: Peter J. Knowles, Nicholas C. Handy,
            A determinant based full configuration interaction program,
            Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
            ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.
    """

    def __init__(self, fcidump_input: str, atoms: Optional[List[str]] = None) -> None:
        """
        Args:
            fcidump_input: Path to the FCIDump file.
            atoms: Allows to specify the atom list of the molecule. If it is provided, the created
                QMolecule instance will permit frozen core Hamiltonians. This list must consist of
                valid atom symbols.

        Raises:
            QiskitChemistryError: If ``fcidump_input`` is not a string or if ``atoms`` is not a list
                of valid atomic symbols as specified in ``QMolecule``.
        """
        super().__init__()

        if not isinstance(fcidump_input, str):
            raise QiskitChemistryError(
                "The fcidump_input must be str, not '{}'".format(fcidump_input))
        self._fcidump_input = fcidump_input

        if atoms and not isinstance(atoms, list) \
                and not all([sym in QMolecule.symbols for sym in atoms]):
            raise QiskitChemistryError(
                "The atoms must be a list of valid atomic symbols, not '{}'".format(atoms))
        self.atoms = atoms

    def run(self) -> QMolecule:
        """Constructs a QMolecule instance out of a FCIDump file.

        Returns:
            A QMolecule instance populated with a minimal set of required data.
        """
        fcidump_data = parse(self._fcidump_input)

        q_mol = QMolecule()

        q_mol.nuclear_repulsion_energy = fcidump_data.get('ecore', None)
        q_mol.num_orbitals = fcidump_data.get('NORB')
        q_mol.multiplicity = fcidump_data.get('MS2', 0) + 1
        q_mol.charge = 0  # ensures QMolecule.log() works
        q_mol.num_beta = (fcidump_data.get('NELEC') - (q_mol.multiplicity - 1)) // 2
        q_mol.num_alpha = fcidump_data.get('NELEC') - q_mol.num_beta
        if self.atoms is not None:
            q_mol.num_atoms = len(self.atoms)
            q_mol.atom_symbol = self.atoms
            q_mol.atom_xyz = [[float('NaN')] * 3] * len(self.atoms)  # ensures QMolecule.log() works

        q_mol.mo_onee_ints = fcidump_data.get('hij', None)
        q_mol.mo_onee_ints_b = fcidump_data.get('hij_b', None)
        q_mol.mo_eri_ints = fcidump_data.get('hijkl', None)
        q_mol.mo_eri_ints_bb = fcidump_data.get('hijkl_bb', None)
        q_mol.mo_eri_ints_ba = fcidump_data.get('hijkl_ba', None)

        return q_mol

    @staticmethod
    def dump(q_mol: QMolecule, outpath: str, orbsym: Optional[List[int]] = None,
             isym: int = 1) -> None:
        """Convenience method to produce an FCIDump output file.

        Args:
            outpath: Path to the output file.
            q_mol: QMolecule data to be dumped. It is assumed that the nuclear_repulsion_energy in
                this QMolecule instance contains the inactive core energy.
            orbsym: A list of spatial symmetries of the orbitals.
            isym: The spatial symmetry of the wave function.
        """
        dump(outpath,
             q_mol.num_orbitals, q_mol.num_alpha + q_mol.num_beta,
             (q_mol.mo_onee_ints, q_mol.mo_onee_ints_b),
             (q_mol.mo_eri_ints, q_mol.mo_eri_ints_ba, q_mol.mo_eri_ints_bb),
             q_mol.nuclear_repulsion_energy, ms2=q_mol.multiplicity - 1, orbsym=orbsym, isym=isym)
