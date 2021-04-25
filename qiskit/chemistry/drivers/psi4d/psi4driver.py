# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" PSI4 Driver """

from typing import Union, List, Optional
import tempfile
import os
import subprocess
import logging
import sys
from shutil import which
from ..units_type import UnitsType
from ..fermionic_driver import FermionicDriver, HFMethodType
from ...qiskit_chemistry_error import QiskitChemistryError
from ..molecule import Molecule
from ...qmolecule import QMolecule

logger = logging.getLogger(__name__)

PSI4 = 'psi4'

PSI4_APP = which(PSI4)


class PSI4Driver(FermionicDriver):
    """
    Qiskit chemistry driver using the PSI4 program.

    See http://www.psicode.org/
    """

    def __init__(self,
                 config: Union[str, List[str]] =
                 'molecule h2 {\n  0 1\n  H  0.0 0.0 0.0\n  H  0.0 0.0 0.735\n}\n\n'
                 'set {\n  basis sto-3g\n  scf_type pk\n  reference rhf\n',
                 molecule: Optional[Molecule] = None,
                 basis: str = 'sto-3g',
                 hf_method: HFMethodType = HFMethodType.RHF) -> None:
        """
        Args:
            config: A molecular configuration conforming to PSI4 format.
            molecule: A driver independent Molecule definition instance may be provided. When
                a molecule is supplied the `config` parameter is ignored and the Molecule instance,
                along with `basis` and `hf_method` is used to build a basic config instead.
                The Molecule object is read when the driver is run and converted to the driver
                dependent configuration for the computation. This allows, for example, the Molecule
                geometry to be updated to compute different points.
            basis: Basis set
            hf_method: Hartree-Fock Method type

        Raises:
            QiskitChemistryError: Invalid Input
        """
        self._check_valid()
        if not isinstance(config, str) and not isinstance(config, list):
            raise QiskitChemistryError("Invalid config for PSI4 Driver '{}'".format(config))

        if isinstance(config, list):
            config = '\n'.join(config)

        super().__init__(molecule=molecule,
                         basis=basis,
                         hf_method=hf_method.value,
                         supports_molecule=True)
        self._config = config

    @staticmethod
    def _check_valid():
        if PSI4_APP is None:
            raise QiskitChemistryError("Could not locate {}".format(PSI4))

    def _from_molecule_to_str(self) -> str:
        units = None
        if self.molecule.units == UnitsType.ANGSTROM:
            units = 'ang'
        elif self.molecule.units == UnitsType.BOHR:
            units = 'bohr'
        else:
            raise QiskitChemistryError("Unknown unit '{}'".format(self.molecule.units.value))
        name = ''.join([name for (name, _) in self.molecule.geometry])
        geom = '\n'.join([name + ' ' + ' '.join(map(str, coord))
                          for (name, coord) in self.molecule.geometry])
        cfg1 = f'molecule {name} {{\nunits {units}\n'
        cfg2 = f'{self.molecule.charge} {self.molecule.multiplicity}\n'
        cfg3 = f'{geom}\nno_com\nno_reorient\n}}\n\n'
        cfg4 = f'set {{\n basis {self.basis}\n scf_type pk\n reference {self.hf_method}\n}}'
        return cfg1 + cfg2 + cfg3 + cfg4

    def run(self) -> QMolecule:
        if self.molecule is not None:
            cfg = self._from_molecule_to_str()
        else:
            cfg = self._config

        psi4d_directory = os.path.dirname(os.path.realpath(__file__))
        template_file = psi4d_directory + '/_template.txt'
        qiskit_chemistry_directory = os.path.abspath(os.path.join(psi4d_directory, '../..'))

        molecule = QMolecule()

        input_text = cfg + '\n'
        input_text += 'import sys\n'
        syspath = '[\'' + qiskit_chemistry_directory + '\',\'' + '\',\''.join(sys.path) + '\']'

        input_text += 'sys.path = ' + syspath + ' + sys.path\n'
        input_text += 'from qmolecule import QMolecule\n'
        input_text += '_q_molecule = QMolecule("{0}")\n'.format(molecule.filename)

        with open(template_file, 'r') as file:
            input_text += file.read()

        file_fd, input_file = tempfile.mkstemp(suffix='.inp')
        os.close(file_fd)
        with open(input_file, 'w') as stream:
            stream.write(input_text)

        file_fd, output_file = tempfile.mkstemp(suffix='.out')
        os.close(file_fd)
        try:
            PSI4Driver._run_psi4(input_file, output_file)
            if logger.isEnabledFor(logging.DEBUG):
                with open(output_file, 'r') as file:
                    logger.debug('PSI4 output file:\n%s', file.read())
        finally:
            run_directory = os.getcwd()
            for local_file in os.listdir(run_directory):
                if local_file.endswith('.clean'):
                    os.remove(run_directory + '/' + local_file)
            try:
                os.remove('timer.dat')
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                os.remove(input_file)
            except Exception:  # pylint: disable=broad-except
                pass

            try:
                os.remove(output_file)
            except Exception:  # pylint: disable=broad-except
                pass

        _q_molecule = QMolecule(molecule.filename)
        _q_molecule.load()
        # remove internal file
        _q_molecule.remove_file()
        _q_molecule.origin_driver_name = 'PSI4'
        _q_molecule.origin_driver_config = cfg
        return _q_molecule

    @staticmethod
    def _run_psi4(input_file, output_file):

        # Run psi4.
        process = None
        try:
            with subprocess.Popen([PSI4, input_file, output_file],
                                  stdout=subprocess.PIPE, universal_newlines=True) as process:
                stdout, _ = process.communicate()
                process.wait()
        except Exception as ex:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(PSI4)) from ex

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                for i, _ in enumerate(lines):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitChemistryError('{} process return code {}\n{}'.format(
                PSI4, process.returncode, errmsg))
