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
This module implements the abstract base class for Pluggable modules.

To create add-on pluggable modules subclass the Pluggable
class in this module.
Doing so requires that the required pluggable interface is implemented.
"""

from abc import ABC, abstractmethod
import logging
import copy
import numpy as np
import jsonschema
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


class Pluggable(ABC):
    """
    Base class for Pluggables.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    """

    CONFIGURATION = None

    @abstractmethod
    def __init__(self):
        self.check_pluggable_valid()
        self._configuration = copy.deepcopy(self.CONFIGURATION)

    @property
    def configuration(self):
        """Return pluggable configuration."""
        return self._configuration

    @staticmethod
    def check_pluggable_valid():
        """Checks if pluggable is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        """ validate input """
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        properties_dict = schema_dict.get('properties', None)
        if properties_dict is None:
            return

        json_dict = {}
        for property_name, _ in properties_dict.items():
            if property_name in args_dict:
                value = args_dict[property_name]
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                json_dict[property_name] = value
        try:
            jsonschema.validate(json_dict, schema_dict)
        except jsonschema.exceptions.ValidationError as vex:
            raise AquaError(vex.message)
