# -*- coding: utf-8 -*-

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

""" Gaussian Driver """

from typing import Union, List
import sys
import io
import logging
import os
from subprocess import Popen, PIPE
from shutil import which
import tempfile
import numpy as np
from qiskit.chemistry import QMolecule, QiskitChemistryError
from qiskit.chemistry.drivers import BaseDriver

logger = logging.getLogger(__name__)

GAUSSIAN_16 = 'g16'
GAUSSIAN_16_DESC = 'Gaussian 16'

G16PROG = which(GAUSSIAN_16)


class GaussianDriver(BaseDriver):
    """
    Qiskit chemistry driver using the Gaussian™ 16 program.

    See http://gaussian.com/gaussian16/

    This driver uses the Gaussian open-source Gaussian 16 interfacing code in
    order to access integrals and other electronic structure information as
    computed by G16 for the given molecule. The job control file, as provided
    here for the molecular configuration, is augmented for our needs here such
    as to have it output a MatrixElement file.
    """

    def __init__(self,
                 config: Union[str, List[str]] =
                 '# rhf/sto-3g scf(conventional)\n\n'
                 'h2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735\n\n') -> None:
        """
        Args:
            config: A molecular configuration conforming to Gaussian™ 16 format
        Raises:
            QiskitChemistryError: Invalid Input
        """
        GaussianDriver._check_valid()
        if not isinstance(config, list) and not isinstance(config, str):
            raise QiskitChemistryError("Invalid input for Gaussian Driver '{}'".format(config))

        if isinstance(config, list):
            config = '\n'.join(config)

        super().__init__()
        self._config = config

    @staticmethod
    def _check_valid():
        if G16PROG is None:
            raise QiskitChemistryError(
                "Could not locate {} executable '{}'. Please check that it is installed correctly."
                .format(GAUSSIAN_16_DESC, GAUSSIAN_16))

    def run(self) -> QMolecule:
        cfg = self._config
        while not cfg.endswith('\n\n'):
            cfg += '\n'

        logger.debug("User supplied configuration raw: '%s'",
                     cfg.replace('\r', '\\r').replace('\n', '\\n'))
        logger.debug('User supplied configuration\n%s', cfg)

        # To the Gaussian section of the input file passed here as section string
        # add line '# Symm=NoInt output=(matrix,i4labels,mo2el) tran=full'
        # NB: Line above needs to be added in right context, i.e after any lines
        #     beginning with % along with any others that start with #
        # append at end the name of the MatrixElement file to be written

        file, fname = tempfile.mkstemp(suffix='.mat')
        os.close(file)

        cfg = self._augment_config(fname, cfg)
        logger.debug('Augmented control information:\n%s', cfg)

        GaussianDriver._run_g16(cfg)

        q_mol = self._parse_matrix_file(fname)
        try:
            os.remove(fname)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to remove MatrixElement file %s", fname)

        q_mol.origin_driver_name = 'GAUSSIAN'
        q_mol.origin_driver_config = self._config
        return q_mol

    # Adds the extra config we need to the input file
    def _augment_config(self, fname, cfg):
        cfgaug = ""
        with io.StringIO() as outf:
            with io.StringIO(cfg) as inf:
                # Add our Route line at the end of any existing ones
                line = ""
                added = False
                while not added:
                    line = inf.readline()
                    if not line:
                        break
                    if line.startswith('#'):
                        outf.write(line)
                        while not added:
                            line = inf.readline()
                            if not line:
                                raise QiskitChemistryError('Unexpected end of Gaussian input')
                            if not line.strip():
                                outf.write('# Window=Full Int=NoRaff Symm=(NoInt,None) '
                                           'output=(matrix,i4labels,mo2el) tran=full\n')
                                added = True
                            outf.write(line)
                    else:
                        outf.write(line)

                # Now add our filename after the title and molecule but
                # before any additional data. We located
                # the end of the # section by looking for a blank line after
                # the first #. Allows comment lines
                # to be inter-mixed with Route lines if that's ever done.
                # From here we need to see two sections
                # more, the title and molecule so we can add the filename.
                added = False
                section_count = 0
                blank = True
                while not added:
                    line = inf.readline()
                    if not line:
                        raise QiskitChemistryError('Unexpected end of Gaussian input')
                    if not line.strip():
                        blank = True
                        if section_count == 2:
                            break
                    else:
                        if blank:
                            section_count += 1
                            blank = False
                    outf.write(line)

                outf.write(line)
                outf.write(fname)
                outf.write('\n\n')

                # Whatever is left in the original config we just append without further inspection
                while True:
                    line = inf.readline()
                    if not line:
                        break
                    outf.write(line)

                cfgaug = outf.getvalue()

        return cfgaug

    def _parse_matrix_file(self, fname, useao2e=False):
        # get_driver_class is used here because the discovery routine will load all the gaussian
        # binary dependencies, if not loaded already. It won't work without it.
        try:
            # add gauopen to sys.path so that binaries can be loaded
            gauopen_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gauopen')
            if gauopen_directory not in sys.path:
                sys.path.insert(0, gauopen_directory)
            # pylint: disable=import-outside-toplevel
            from .gauopen.QCMatEl import MatEl
        except ImportError as mnfe:
            msg = ('qcmatrixio extension not found. '
                   'See Gaussian driver readme to build qcmatrixio.F using f2py') \
                if mnfe.name == 'qcmatrixio' else str(mnfe)

            logger.info(msg)
            raise QiskitChemistryError(msg) from mnfe

        mel = MatEl(file=fname)
        logger.debug('MatrixElement file:\n%s', mel)

        # Create driver level molecule object and populate
        _q_ = QMolecule()
        _q_.origin_driver_version = mel.gversion
        # Energies and orbits
        _q_.hf_energy = mel.scalar('ETOTAL')
        _q_.nuclear_repulsion_energy = mel.scalar('ENUCREP')
        _q_.num_orbitals = 0  # updated below from orbital coeffs size
        _q_.num_alpha = (mel.ne + mel.multip - 1) // 2
        _q_.num_beta = (mel.ne - mel.multip + 1) // 2
        moc = self._get_matrix(mel, 'ALPHA MO COEFFICIENTS')
        moc_b = self._get_matrix(mel, 'BETA MO COEFFICIENTS')
        if np.array_equal(moc, moc_b):
            logger.debug('ALPHA and BETA MO COEFFS identical, keeping only ALPHA')
            moc_b = None
        _q_.num_orbitals = moc.shape[0]
        _q_.mo_coeff = moc
        _q_.mo_coeff_b = moc_b
        orbs_energy = self._get_matrix(mel, 'ALPHA ORBITAL ENERGIES')
        _q_.orbital_energies = orbs_energy
        orbs_energy_b = self._get_matrix(mel, 'BETA ORBITAL ENERGIES')
        _q_.orbital_energies_b = orbs_energy_b if moc_b is not None else None
        # Molecule geometry
        _q_.molecular_charge = mel.icharg
        _q_.multiplicity = mel.multip
        _q_.num_atoms = mel.natoms
        _q_.atom_symbol = []
        _q_.atom_xyz = np.empty([mel.natoms, 3])
        syms = mel.ian
        xyz = np.reshape(mel.c, (_q_.num_atoms, 3))
        for n_i in range(0, _q_.num_atoms):
            _q_.atom_symbol.append(QMolecule.symbols[syms[n_i]])
            for idx in range(xyz.shape[1]):
                coord = xyz[n_i][idx]
                if abs(coord) < 1e-10:
                    coord = 0
                _q_.atom_xyz[n_i][idx] = coord

        # 1 and 2 electron integrals
        hcore = self._get_matrix(mel, 'CORE HAMILTONIAN ALPHA')
        logger.debug('CORE HAMILTONIAN ALPHA %s', hcore.shape)
        hcore_b = self._get_matrix(mel, 'CORE HAMILTONIAN BETA')
        if np.array_equal(hcore, hcore_b):
            # From Gaussian interfacing documentation: "The two
            # core Hamiltonians are identical unless
            # a Fermi contact perturbation has been applied."
            logger.debug('CORE HAMILTONIAN ALPHA and BETA identical, keeping only ALPHA')
            hcore_b = None
        logger.debug('CORE HAMILTONIAN BETA %s',
                     '- Not present' if hcore_b is None else hcore_b.shape)
        kinetic = self._get_matrix(mel, 'KINETIC ENERGY')
        logger.debug('KINETIC ENERGY %s', kinetic.shape)
        overlap = self._get_matrix(mel, 'OVERLAP')
        logger.debug('OVERLAP %s', overlap.shape)
        mohij = QMolecule.oneeints2mo(hcore, moc)
        mohij_b = None
        if moc_b is not None:
            mohij_b = QMolecule.oneeints2mo(hcore if hcore_b is None else hcore_b, moc_b)

        eri = self._get_matrix(mel, 'REGULAR 2E INTEGRALS')
        logger.debug('REGULAR 2E INTEGRALS %s', eri.shape)
        if moc_b is None and mel.matlist.get('BB MO 2E INTEGRALS') is not None:
            # It seems that when using ROHF, where alpha and beta coeffs are
            # the same, that integrals
            # for BB and BA are included in the output, as well as just AA
            # that would have been expected
            # Using these fails to give the right answer (is ok for UHF).
            # So in this case we revert to
            # using 2 electron ints in atomic basis from the output and
            # converting them ourselves.
            useao2e = True
            logger.info(
                'Identical A and B coeffs but BB ints are present - using regular 2E ints instead')

        if useao2e:
            # eri are 2-body in AO. We can convert to MO via the QMolecule
            # method but using ints in MO already, as in the else here, is better
            mohijkl = QMolecule.twoeints2mo(eri, moc)
            mohijkl_bb = None
            mohijkl_ba = None
            if moc_b is not None:
                mohijkl_bb = QMolecule.twoeints2mo(eri, moc_b)
                mohijkl_ba = QMolecule.twoeints2mo_general(eri, moc_b, moc_b, moc, moc)
        else:
            # These are in MO basis but by default will be reduced in size by
            # frozen core default so to use them we need to add Window=Full
            # above when we augment the config
            mohijkl = self._get_matrix(mel, 'AA MO 2E INTEGRALS')
            logger.debug('AA MO 2E INTEGRALS %s', mohijkl.shape)
            mohijkl_bb = self._get_matrix(mel, 'BB MO 2E INTEGRALS')
            logger.debug('BB MO 2E INTEGRALS %s',
                         '- Not present' if mohijkl_bb is None else mohijkl_bb.shape)
            mohijkl_ba = self._get_matrix(mel, 'BA MO 2E INTEGRALS')
            logger.debug('BA MO 2E INTEGRALS %s',
                         '- Not present' if mohijkl_ba is None else mohijkl_ba.shape)

        _q_.hcore = hcore
        _q_.hcore_b = hcore_b
        _q_.kinetic = kinetic
        _q_.overlap = overlap
        _q_.eri = eri

        _q_.mo_onee_ints = mohij
        _q_.mo_onee_ints_b = mohij_b
        _q_.mo_eri_ints = mohijkl
        _q_.mo_eri_ints_bb = mohijkl_bb
        _q_.mo_eri_ints_ba = mohijkl_ba

        # dipole moment
        dipints = self._get_matrix(mel, 'DIPOLE INTEGRALS')
        dipints = np.einsum('ijk->kji', dipints)
        _q_.x_dip_ints = dipints[0]
        _q_.y_dip_ints = dipints[1]
        _q_.z_dip_ints = dipints[2]
        _q_.x_dip_mo_ints = QMolecule.oneeints2mo(dipints[0], moc)
        _q_.x_dip_mo_ints_b = None
        _q_.y_dip_mo_ints = QMolecule.oneeints2mo(dipints[1], moc)
        _q_.y_dip_mo_ints_b = None
        _q_.z_dip_mo_ints = QMolecule.oneeints2mo(dipints[2], moc)
        _q_.z_dip_mo_ints_b = None
        if moc_b is not None:
            _q_.x_dip_mo_ints_b = QMolecule.oneeints2mo(dipints[0], moc_b)
            _q_.y_dip_mo_ints_b = QMolecule.oneeints2mo(dipints[1], moc_b)
            _q_.z_dip_mo_ints_b = QMolecule.oneeints2mo(dipints[2], moc_b)

        nucl_dip = np.einsum('i,ix->x', syms, xyz)
        nucl_dip = np.round(nucl_dip, decimals=8)
        _q_.nuclear_dipole_moment = nucl_dip
        _q_.reverse_dipole_sign = True

        return _q_

    def _get_matrix(self, mel, name):
        # Gaussian dimens values may be negative which it itself handles in expand
        # but convert to all positive for use in reshape. Note: Fortran index ordering.
        m_x = mel.matlist.get(name)
        if m_x is None:
            return None
        dims = tuple([abs(i) for i in m_x.dimens])
        mat = np.reshape(m_x.expand(), dims, order='F')
        return mat

    @staticmethod
    def _run_g16(cfg):

        # Run Gaussian 16. We capture stdout and if error log the last 10 lines that
        # should include the error description from Gaussian
        process = None
        try:
            process = Popen(GAUSSIAN_16, stdin=PIPE, stdout=PIPE, universal_newlines=True)
            stdout, _ = process.communicate(cfg)
            process.wait()
        except Exception as ex:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(GAUSSIAN_16_DESC)) from ex

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                start = 0
                if len(lines) > 10:
                    start = len(lines) - 10
                for i in range(start, len(lines)):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitChemistryError(
                '{} process return code {}\n{}'.format(
                    GAUSSIAN_16_DESC, process.returncode, errmsg))

        if logger.isEnabledFor(logging.DEBUG):
            alltext = ""
            if stdout is not None:
                lines = stdout.splitlines()
                for line in lines:
                    alltext += line + "\n"
            logger.debug("Gaussian output:\n%s", alltext)
