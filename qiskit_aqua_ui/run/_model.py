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

from .base_model import BaseModel
import os
from qiskit_aqua_ui._uipreferences import UIPreferences
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class Model(BaseModel):

    def __init__(self):
        """Create Model object."""
        super().__init__()

    def new(self):
        from qiskit.aqua.parser._inputparser import InputParser
        uipreferences = UIPreferences()
        return super().new(InputParser,
                           os.path.join(os.path.dirname(__file__), 'input_template.json'),
                           uipreferences.get_populate_defaults(True))

    def load_file(self, filename):
        from qiskit.aqua.parser._inputparser import InputParser
        uipreferences = UIPreferences()
        return super().load_file(filename, InputParser, uipreferences.get_populate_defaults(True))

    def default_properties_equals_properties(self, section_name):
        from qiskit.aqua.parser import JSONSchema
        if self.section_is_text(section_name):
            return self.get_section_default_properties(section_name) == self.get_section_text(section_name)

        default_properties = self.get_section_default_properties(section_name)
        if isinstance(default_properties, OrderedDict):
            default_properties = dict(default_properties)

        properties = self.get_section_properties(section_name)
        if isinstance(properties, OrderedDict):
            properties = dict(properties)

        if not isinstance(default_properties, dict) or not isinstance(properties, dict):
            return default_properties == properties

        if JSONSchema.BACKEND != section_name and JSONSchema.NAME in properties:
            default_properties[JSONSchema.NAME] = properties[JSONSchema.NAME]

        return default_properties == properties

    def get_input_section_names(self):
        from qiskit.aqua.parser._inputparser import InputParser
        from qiskit.aqua import local_pluggables, PluggableType
        from qiskit.aqua.parser import JSONSchema
        problem_name = None
        if self._parser is not None:
            problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            return local_pluggables(PluggableType.INPUT)

        input_names = []
        for input_name in local_pluggables(PluggableType.INPUT):
            problems = InputParser.get_input_problems(input_name)
            if problem_name in problems:
                input_names.append(input_name)

        return input_names
