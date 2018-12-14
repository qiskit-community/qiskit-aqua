# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
This module implements the abstract base class for Pluggable modules.

To create add-on pluggable modules subclass the Pluggable
class in this module.
Doing so requires that the required pluggable interface is implemented.
"""

from abc import ABC, abstractmethod
import logging
import copy
from qiskit_aqua.parser import JSONSchema

logger = logging.getLogger(__name__)


class Pluggable(ABC):
    """
    Base class for Pluggables.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    """
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
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        jsonSchema = JSONSchema(schema_dict)
        schema_property_names = jsonSchema.get_default_section_names()
        json_dict = {}
        for property_name in schema_property_names:
            if property_name in args_dict:
                json_dict[property_name] = args_dict[property_name]

        jsonSchema.validate(json_dict)
