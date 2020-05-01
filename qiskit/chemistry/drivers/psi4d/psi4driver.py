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

""" PSI4 Driver """

import tempfile
import os
import subprocess
import logging
import sys
from shutil import which
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry import QMolecule, QiskitChemistryError

logger = logging.getLogger(__name__)

PSI4 = 'psi4'

PSI4_APP = which(PSI4)


class PSI4Driver(BaseDriver):
    """Python implementation of a psi4 driver."""

    CONFIGURATION = {
        "name": "PSI4",
        "description": "PSI4 Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "id": "psi4_schema",
            "type": "string",
            "default": "molecule h2 {\n  0 1\n  H  0.0 0.0 0.0\n  H  0.0 0.0 0.735\n}\n\n"
                       "set {\n  basis sto-3g\n  scf_type pk\n  reference rhf\n}"
        }
    }

    def __init__(self, config):
        """
        Initializer
        Args:
            config (str or list): driver configuration
        Raises:
            QiskitChemistryError: Invalid Input
        """
        if not isinstance(config, list) and not isinstance(config, str):
            raise QiskitChemistryError("Invalid input for PSI4 Driver '{}'".format(config))

        if isinstance(config, list):
            config = '\n'.join(config)

        super().__init__()
        self._config = config

    @staticmethod
    def check_driver_valid():
        if PSI4_APP is None:
            raise QiskitChemistryError("Could not locate {}".format(PSI4))

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            PSI4Driver: Driver object
        Raises:
            QiskitChemistryError: Invalid or missing section
        """
        if not isinstance(section, str):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        kwargs = {'config': section}
        logger.debug('init_from_input: %s', kwargs)
        return cls(**kwargs)

    def run(self):
        # create input
        psi4d_directory = os.path.dirname(os.path.realpath(__file__))
        template_file = psi4d_directory + '/_template.txt'
        qiskit_chemistry_directory = os.path.abspath(os.path.join(psi4d_directory, '../..'))

        molecule = QMolecule()

        input_text = self._config + '\n'
        input_text += 'import sys\n'
        syspath = '[\'' + qiskit_chemistry_directory + '\',\'' + '\',\''.join(sys.path) + '\']'

        input_text += 'sys.path = ' + syspath + ' + sys.path\n'
        input_text += 'from qmolecule import QMolecule\n'
        input_text += '_q_molecule = QMolecule("{0}")\n'.format(molecule.filename)

        with open(template_file, 'r') as file:
            input_text += file.read()

        file, input_file = tempfile.mkstemp(suffix='.inp')
        os.close(file)
        with open(input_file, 'w') as stream:
            stream.write(input_text)

        file, output_file = tempfile.mkstemp(suffix='.out')
        os.close(file)
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
        _q_molecule.origin_driver_name = self.configuration['name']
        _q_molecule.origin_driver_config = self._config
        return _q_molecule

    @staticmethod
    def _run_psi4(input_file, output_file):

        # Run psi4.
        process = None
        try:
            process = subprocess.Popen([PSI4, input_file, output_file],
                                       stdout=subprocess.PIPE, universal_newlines=True)
            stdout, _ = process.communicate()
            process.wait()
        except Exception:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(PSI4))

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                for i, _ in enumerate(lines):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitChemistryError('{} process return code {}\n{}'.format(
                PSI4, process.returncode, errmsg))
