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

"""
This module implements the abstract base class for driver modules.

To create add-on driver modules subclass the BaseDriver class in this module.
Doing so requires that the required driver interface is implemented.
"""

from abc import ABC, abstractmethod
import copy
from enum import Enum
import logging
import jsonschema

logger = logging.getLogger(__name__)


class UnitsType(Enum):
    """ Units Type Enum """
    ANGSTROM = 'Angstrom'
    BOHR = 'Bohr'


class HFMethodType(Enum):
    """ HFMethodType Enum """
    RHF = 'rhf'
    ROHF = 'rohf'
    UHF = 'uhf'


class BaseDriver(ABC):
    """
    Base class for Drivers.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    """

    CONFIGURATION = None

    @abstractmethod
    def __init__(self):
        self.check_driver_valid()
        self._configuration = copy.deepcopy(self.CONFIGURATION)
        self._work_path = None

    @property
    def configuration(self):
        """Return driver configuration."""
        return self._configuration

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            section (dict): section dictionary

        Returns:
            BaseDriver: Driver object
        """
        pass

    @staticmethod
    def check_driver_valid():
        """Checks if driver is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        """ validate driver input """
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        properties_dict = schema_dict.get('properties', None)
        if properties_dict is None:
            return

        json_dict = {}
        for property_name, _ in properties_dict.items():
            if property_name in args_dict:
                json_dict[property_name] = args_dict[property_name]

        jsonschema.validate(json_dict, schema_dict)

    @property
    def work_path(self):
        """ returns work path """
        return self._work_path

    @work_path.setter
    def work_path(self, new_work_path):
        """ sets work path """
        self._work_path = new_work_path

    @abstractmethod
    def run(self):
        """ runs driver """
        pass
