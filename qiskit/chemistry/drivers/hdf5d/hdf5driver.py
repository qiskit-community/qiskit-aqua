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

""" HDF5 Driver """

import os
import logging
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry import QMolecule, QiskitChemistryError

logger = logging.getLogger(__name__)


class HDF5Driver(BaseDriver):
    """Python implementation of a hdf5 driver."""

    KEY_HDF5_INPUT = 'hdf5_input'

    CONFIGURATION = {
        "name": "HDF5",
        "description": "HDF5 Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "id": "hdf5_schema",
            "type": "object",
            "properties": {
                "hdf5_input": {
                    "type": "string",
                    "default": "molecule.hdf5"
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self, hdf5_input):
        """
        Initializer
        Args:
            hdf5_input (str): path to hdf5 file
        """
        self.validate(locals())
        super().__init__()
        self._hdf5_input = hdf5_input

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            HDF5Driver: Driver object
        Raises:
            QiskitChemistryError: Invalid or missing section
        """
        if section is None or not isinstance(section, dict):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        kwargs = section
        logger.debug('init_from_input: %s', kwargs)
        return cls(**kwargs)

    def run(self):
        hdf5_file = self._hdf5_input
        if self.work_path is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(self.work_path, hdf5_file))

        if not os.path.isfile(hdf5_file):
            raise LookupError('HDF5 file not found: {}'.format(hdf5_file))

        molecule = QMolecule(hdf5_file)
        molecule.load()
        return molecule
