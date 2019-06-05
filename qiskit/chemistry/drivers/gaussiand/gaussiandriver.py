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
import io
import logging
import os
from subprocess import Popen, PIPE
from shutil import which
import tempfile
import numpy as np
import sys
from qiskit.chemistry import QMolecule, QiskitChemistryError
from qiskit.chemistry.drivers import BaseDriver

logger = logging.getLogger(__name__)

GAUSSIAN_16 = 'g16'
GAUSSIAN_16_DESC = 'Gaussian 16'

g16prog = which(GAUSSIAN_16)


class GaussianDriver(BaseDriver):
    """Python implementation of a Gaussian 16 driver.

    This driver uses the Gaussian open-source Gaussian 16 interfacing code in
    order to access integrals and other electronic structure information as
    computed by G16 for the given molecule. The job control file, as provided
    via our input file, is augmented for our needs here such as to have it
    output a MatrixElement file.
    """

    CONFIGURATION = {
        "name": "GAUSSIAN",
        "description": "Gaussian 16 Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "gaussian_schema",
            "type": "string",
            "default": "# rhf/sto-3g scf(conventional)\n\nh2 molecule\n\n0 1\nH   0.0  0.0    0.0\nH   0.0  0.0    0.735\n\n"
        }
    }

    def __init__(self, config):
        """
        Initializer
        Args:
            config (str or list): driver configuration
        """
        if not isinstance(config, list) and not isinstance(config, str):
            raise QiskitChemistryError("Invalid input for Gaussian Driver '{}'".format(config))

        if isinstance(config, list):
            config = '\n'.join(config)

        super().__init__()
        self._config = config

    @staticmethod
    def check_driver_valid():
        if g16prog is None:
            raise QiskitChemistryError("Could not locate {} executable '{}'. Please check that it is installed correctly."
                                       .format(GAUSSIAN_16_DESC, GAUSSIAN_16))

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        if not isinstance(section, str):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        kwargs = {'config': section}
        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        cfg = self._config
        while not cfg.endswith('\n\n'):
            cfg += '\n'

        logger.debug("User supplied configuration raw: '{}'".format(cfg.replace('\r', '\\r').replace('\n', '\\n')))
        logger.debug('User supplied configuration\n{}'.format(cfg))

        # To the Gaussian section of the input file passed here as section string
        # add line '# Symm=NoInt output=(matrix,i4labels,mo2el) tran=full'
        # NB: Line above needs to be added in right context, i.e after any lines
        #     beginning with % along with any others that start with #
        # append at end the name of the MatrixElement file to be written

        fd, fname = tempfile.mkstemp(suffix='.mat')
        os.close(fd)

        cfg = self._augment_config(fname, cfg)
        logger.debug('Augmented control information:\n{}'.format(cfg))

        GaussianDriver._run_g16(cfg)

        q_mol = self._parse_matrix_file(fname)
        try:
            os.remove(fname)
        except:
            logger.warning("Failed to remove MatrixElement file " + fname)

        q_mol.origin_driver_name = self.configuration['name']
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
                            if len(line.strip()) == 0:
                                outf.write('# Window=Full Int=NoRaff Symm=(NoInt,None) output=(matrix,i4labels,mo2el) tran=full\n')
                                added = True
                            outf.write(line)
                    else:
                        outf.write(line)

                # Now add our filename after the title and molecule but before any additional data. We located
                # the end of the # section by looking for a blank line after the first #. Allows comment lines
                # to be inter-mixed with Route lines if that's ever done. From here we need to see two sections
                # more, the title and molecule so we can add the filename.
                added = False
                section_count = 0
                blank = True
                while not added:
                    line = inf.readline()
                    if not line:
                        raise QiskitChemistryError('Unexpected end of Gaussian input')
                    if len(line.strip()) == 0:
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

    def _parse_matrix_file(self, fname, useAO2E=False):
        # get_driver_class is used here because the discovery routine will load all the gaussian
        # binary dependencies, if not loaded already. It won't work without it.
        try:
            # add gauopen to sys.path so that binaries can be loaded
            gauopen_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gauopen')
            if gauopen_directory not in sys.path:
                sys.path.insert(0, gauopen_directory)

            from .gauopen.QCMatEl import MatEl
        except ImportError as mnfe:
            msg = 'qcmatrixio extension not found. See Gaussian driver readme to build qcmatrixio.F using f2py' \
                if mnfe.name == 'qcmatrixio' else str(mnfe)

            logger.info(msg)
            raise QiskitChemistryError(msg)

        mel = MatEl(file=fname)
        logger.debug('MatrixElement file:\n{}'.format(mel))

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
        moc_B = self._get_matrix(mel, 'BETA MO COEFFICIENTS')
        if np.array_equal(moc, moc_B):
            logger.debug('ALPHA and BETA MO COEFFS identical, keeping only ALPHA')
            moc_B = None
        _q_.num_orbitals = moc.shape[0]
        _q_.mo_coeff = moc
        _q_.mo_coeff_B = moc_B
        orbs_energy = self._get_matrix(mel, 'ALPHA ORBITAL ENERGIES')
        _q_.orbital_energies = orbs_energy
        orbs_energy_B = self._get_matrix(mel, 'BETA ORBITAL ENERGIES')
        _q_.orbital_energies_B = orbs_energy_B
        # Molecule geometry
        _q_.molecular_charge = mel.icharg
        _q_.multiplicity = mel.multip
        _q_.num_atoms = mel.natoms
        _q_.atom_symbol = []
        _q_.atom_xyz = np.empty([mel.natoms, 3])
        syms = mel.ian
        xyz = np.reshape(mel.c, (_q_.num_atoms, 3))
        for _n in range(0, _q_.num_atoms):
            _q_.atom_symbol.append(QMolecule.symbols[syms[_n]])
            for _i in range(xyz.shape[1]):
                coord = xyz[_n][_i]
                if abs(coord) < 1e-10:
                    coord = 0
                _q_.atom_xyz[_n][_i] = coord

        # 1 and 2 electron integrals
        hcore = self._get_matrix(mel, 'CORE HAMILTONIAN ALPHA')
        logger.debug('CORE HAMILTONIAN ALPHA {}'.format(hcore.shape))
        hcore_B = self._get_matrix(mel, 'CORE HAMILTONIAN BETA')
        if np.array_equal(hcore, hcore_B):
            # From Gaussian interfacing documentation: "The two core Hamiltonians are identical unless
            # a Fermi contact perturbation has been applied."
            logger.debug('CORE HAMILTONIAN ALPHA and BETA identical, keeping only ALPHA')
            hcore_B = None
        logger.debug('CORE HAMILTONIAN BETA {}'.format('- Not present' if hcore_B is None else hcore_B.shape))
        kinetic = self._get_matrix(mel, 'KINETIC ENERGY')
        logger.debug('KINETIC ENERGY {}'.format(kinetic.shape))
        overlap = self._get_matrix(mel, 'OVERLAP')
        logger.debug('OVERLAP {}'.format(overlap.shape))
        mohij = QMolecule.oneeints2mo(hcore, moc)
        mohij_B = None
        if moc_B is not None:
            mohij_B = QMolecule.oneeints2mo(hcore if hcore_B is None else hcore_B, moc_B)

        eri = self._get_matrix(mel, 'REGULAR 2E INTEGRALS')
        logger.debug('REGULAR 2E INTEGRALS {}'.format(eri.shape))
        if useAO2E:
            # eri are 2-body in AO. We can convert to MO via the QMolecule
            # method but using ints in MO already, as in the else here, is better
            mohijkl = QMolecule.twoeints2mo(eri, moc)
            mohijkl_BB = None
            mohijkl_BA = None
        else:
            # These are in MO basis but by default will be reduced in size by
            # frozen core default so to use them we need to add Window=Full
            # above when we augment the config
            mohijkl = self._get_matrix(mel, 'AA MO 2E INTEGRALS')
            logger.debug('AA MO 2E INTEGRALS {}'.format(mohijkl.shape))
            mohijkl_BB = self._get_matrix(mel, 'BB MO 2E INTEGRALS')
            logger.debug('BB MO 2E INTEGRALS {}'.format('- Not present' if mohijkl_BB is None else mohijkl_BB.shape))
            mohijkl_BA = self._get_matrix(mel, 'BA MO 2E INTEGRALS')
            logger.debug('BA MO 2E INTEGRALS {}'.format('- Not present' if mohijkl_BA is None else mohijkl_BA.shape))

        _q_.hcore = hcore
        _q_.hcore_B = hcore_B
        _q_.kinetic = kinetic
        _q_.overlap = overlap
        _q_.eri = eri

        _q_.mo_onee_ints = mohij
        _q_.mo_onee_ints_B = mohij_B
        _q_.mo_eri_ints = mohijkl
        _q_.mo_eri_ints_BB = mohijkl_BB
        _q_.mo_eri_ints_BA = mohijkl_BA

        # dipole moment
        dipints = self._get_matrix(mel, 'DIPOLE INTEGRALS')
        dipints = np.einsum('ijk->kji', dipints)
        _q_.x_dip_ints = dipints[0]
        _q_.y_dip_ints = dipints[1]
        _q_.z_dip_ints = dipints[2]
        _q_.x_dip_mo_ints = QMolecule.oneeints2mo(dipints[0], moc)
        _q_.x_dip_mo_ints_B = None
        _q_.y_dip_mo_ints = QMolecule.oneeints2mo(dipints[1], moc)
        _q_.y_dip_mo_ints_B = None
        _q_.z_dip_mo_ints = QMolecule.oneeints2mo(dipints[2], moc)
        _q_.z_dip_mo_ints_B = None
        if moc_B is not None:
            _q_.x_dip_mo_ints_B = QMolecule.oneeints2mo(dipints[0], moc_B)
            _q_.y_dip_mo_ints_B = QMolecule.oneeints2mo(dipints[1], moc_B)
            _q_.z_dip_mo_ints_N = QMolecule.oneeints2mo(dipints[2], moc_B)

        nucl_dip = np.einsum('i,ix->x', syms, xyz)
        nucl_dip = np.round(nucl_dip, decimals=8)
        _q_.nuclear_dipole_moment = nucl_dip
        _q_.reverse_dipole_sign = True

        return _q_

    def _get_matrix(self, mel, name):
        # Gaussian dimens values may be negative which it itself handles in expand
        # but convert to all positive for use in reshape. Note: Fortran index ordering.
        mx = mel.matlist.get(name)
        if mx is None:
            return None
        dims = tuple([abs(i) for i in mx.dimens])
        mat = np.reshape(mx.expand(), dims, order='F')
        return mat

    @staticmethod
    def _run_g16(cfg):

        # Run Gaussian 16. We capture stdout and if error log the last 10 lines that
        # should include the error description from Gaussian
        process = None
        try:
            process = Popen(GAUSSIAN_16, stdin=PIPE, stdout=PIPE, universal_newlines=True)
            stdout, stderr = process.communicate(cfg)
            process.wait()
        except:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(GAUSSIAN_16_DESC))

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
            raise QiskitChemistryError('{} process return code {}\n{}'.format(GAUSSIAN_16_DESC, process.returncode, errmsg))
        else:
            if logger.isEnabledFor(logging.DEBUG):
                alltext = ""
                if stdout is not None:
                    lines = stdout.splitlines()
                    for line in lines:
                        alltext += line + "\n"
                logger.debug("Gaussian output:\n{}".format(alltext))
