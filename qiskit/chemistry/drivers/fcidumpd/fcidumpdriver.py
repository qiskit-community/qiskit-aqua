# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" FCIDump Driver """

from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry import QiskitChemistryError, QMolecule
from qiskit.chemistry.drivers.fcidumpd.dumper import dump
from qiskit.chemistry.drivers.fcidumpd.parser import parse


class FCIDumpDriver(BaseDriver):
    """
    Python implementation of an FCIDump driver.

    The FCIDump format is partially defined in Knowles1989.

    Knowles1989:
        Peter J. Knowles, Nicholas C. Handy,
        A determinant based full configuration interaction program,
        Computer Physics Communications, Volume 54, Issue 1, 1989, Pages 75-83,
        ISSN 0010-4655, https://doi.org/10.1016/0010-4655(89)90033-7.
    """

    def __init__(self, fcidump_input: str) -> None:
        """
        Initializer

        Args:
            fcidump_input: path to the FCIDump file

        Raises:
            QiskitChemistryError: invalid input
        """
        super().__init__()

        if not isinstance(fcidump_input, str):
            raise QiskitChemistryError("Invalid input for FCIDumpDriver '{}'".format(fcidump_input))
        self._fcidump_input = fcidump_input

    def run(self) -> QMolecule:
        """
        Constructs a QMolecule instance out of a FCIDump file.

        Returns:
            QMolecule: a QMolecule instance populated with a minimal set of required data
        """
        fcidump_data = parse(self._fcidump_input)

        q_mol = QMolecule()

        q_mol.hf_energy = fcidump_data.get('ecore', float('NaN'))
        q_mol.num_orbitals = fcidump_data.get('NORB', float('NaN'))
        # TODO: NELEC is inconclusive in the case of a non-singlet spin system
        q_mol.num_beta = fcidump_data.get('NELEC', float('NaN')) // 2
        q_mol.num_alpha = fcidump_data.get('NELEC', float('NaN')) - q_mol.num_beta

        q_mol.mo_onee_ints = fcidump_data.get('hij', None)
        q_mol.mo_onee_ints_b = fcidump_data.get('hij_b', None)
        q_mol.mo_eri_ints = fcidump_data.get('hijkl', None)
        q_mol.mo_eri_ints_bb = fcidump_data.get('hijkl_bb', None)
        q_mol.mo_eri_ints_ba = fcidump_data.get('hijkl_ba', None)

        return q_mol

    @staticmethod
    def dump(q_mol: QMolecule, outpath: str) -> None:
        """
        Convenience method to produce an FCIDump output file

        Args:
            q_mol (QMolecule): QMolecule data to be dumped. It is assumed that the HF energy stored
            in this QMolecule instance contains the inactive core energy.
            outpath (str): path to the output file
        """
        dump(q_mol.num_orbitals, q_mol.num_alpha + q_mol.num_beta,
             (q_mol.mo_onee_ints, q_mol.mo_onee_ints_b),
             (q_mol.mo_eri_ints, q_mol.mo_eri_ints_ba, q_mol.mo_eri_ints_bb),
             q_mol.hf_energy, outpath)
