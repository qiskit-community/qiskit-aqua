# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM Corp. 2017 and later.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.chemistry.drivers import BaseDriver
import tempfile
import os
import subprocess
import logging
from qiskit.chemistry import QMolecule, QiskitChemistryError
import sys
from shutil import which

logger = logging.getLogger(__name__)

PSI4 = 'psi4'

psi4 = which(PSI4)


class PSI4Driver(BaseDriver):
    """Python implementation of a psi4 driver."""

    CONFIGURATION = {
        "name": "PSI4",
        "description": "PSI4 Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "psi4_schema",
            "type": "string",
            "default": "molecule h2 {\n  0 1\n  H  0.0 0.0 0.0\n  H  0.0 0.0 0.735\n}\n\nset {\n  basis sto-3g\n  scf_type pk\n}"
        }
    }

    def __init__(self, config):
        """
        Initializer
        Args:
            config (str or list): driver configuration
        """
        if not isinstance(config, list) and not isinstance(config, str):
            raise QiskitChemistryError("Invalid input for PSI4 Driver '{}'".format(config))

        if isinstance(config, list):
            config = '\n'.join(config)

        super().__init__()
        self._config = config

    @staticmethod
    def check_driver_valid():
        if psi4 is None:
            raise QiskitChemistryError("Could not locate {}".format(PSI4))

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

        with open(template_file, 'r') as f:
            input_text += f.read()

        fd, input_file = tempfile.mkstemp(suffix='.inp')
        os.close(fd)
        with open(input_file, 'w') as stream:
            stream.write(input_text)

        fd, output_file = tempfile.mkstemp(suffix='.out')
        os.close(fd)
        try:
            PSI4Driver._run_psi4(input_file, output_file)
            if logger.isEnabledFor(logging.DEBUG):
                with open(output_file, 'r') as f:
                    logger.debug('PSI4 output file:\n{}'.format(f.read()))
        finally:
            run_directory = os.getcwd()
            for local_file in os.listdir(run_directory):
                if local_file.endswith('.clean'):
                    os.remove(run_directory + '/' + local_file)
            try:
                os.remove('timer.dat')
            except:
                pass

            try:
                os.remove(input_file)
            except:
                pass

            try:
                os.remove(output_file)
            except:
                pass

        _q_molecule = QMolecule(molecule.filename)
        _q_molecule.load()
        # remove internal file
        _q_molecule.remove_file()
        return _q_molecule

    @staticmethod
    def _run_psi4(input_file, output_file):

        # Run psi4.
        process = None
        try:
            process = subprocess.Popen([PSI4, input_file, output_file],
                                       stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            process.wait()
        except:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(PSI4))

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                for i in range(len(lines)):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitChemistryError('{} process return code {}\n{}'.format(PSI4, process.returncode, errmsg))
