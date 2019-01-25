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

from qiskit_aqua_ui import BaseModel
import os
from ._uipreferences import UIPreferences
import logging

logger = logging.getLogger(__name__)


class Model(BaseModel):

    def __init__(self):
        """Create Model object."""
        super().__init__()

    def new(self):
        from qiskit_chemistry.parser import InputParser
        uipreferences = UIPreferences()
        return super().new(InputParser,
                           os.path.join(os.path.dirname(__file__), 'input_template.json'),
                           uipreferences.get_populate_defaults(True))

    def load_file(self, filename):
        from qiskit_chemistry.parser import InputParser
        uipreferences = UIPreferences()
        return super().load_file(filename, InputParser, uipreferences.get_populate_defaults(True))

    def default_properties_equals_properties(self, section_name):
        from qiskit_aqua.parser import JSONSchema
        if self.section_is_text(section_name):
            return self.get_section_default_properties(section_name) == self._parser.get_section_text(section_name)

        default_properties = self.get_section_default_properties(section_name)
        properties = self.get_section_properties(section_name)
        if not isinstance(default_properties, dict) or not isinstance(properties, dict):
            return default_properties == properties

        if JSONSchema.BACKEND != section_name and JSONSchema.NAME in properties:
            default_properties[JSONSchema.NAME] = properties[JSONSchema.NAME]

        if len(default_properties) != len(properties):
            return False

        substitution_tuples = self._parser.check_if_substitution_key(section_name, list(properties.keys()))
        for substitution_tuple in substitution_tuples:
            property_name = substitution_tuple[0]
            if property_name not in default_properties:
                return False

            if not substitution_tuple[1]:
                if default_properties[property_name] != properties[property_name]:
                    return False

        return True

    def get_dictionary(self):
        if self.is_empty():
            raise Exception("Empty input data.")

        return self._parser.to_dictionary()

    def export_dictionary(self, filename):
        if self.is_empty():
            raise Exception("Empty input data.")

        self._parser.export_dictionary(filename)

    def get_section_properties_with_substitution(self, section_name):
        properties = self.get_section_properties(section_name)
        result_tuples = self._parser.check_if_substitution_key(
            section_name, list(properties.keys()))
        properties_with_substitution = {}
        for result_tuple in result_tuples:
            properties_with_substitution[result_tuple[0]] = (
                properties[result_tuple[0]], result_tuple[1])

        return properties_with_substitution

    def get_operator_section_names(self):
        from qiskit_chemistry.parser import InputParser
        from qiskit_aqua.parser import JSONSchema
        from qiskit_chemistry.core import local_chemistry_operators
        problem_name = None
        if self._parser is not None:
            problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            return local_chemistry_operators()

        operator_names = []
        for operator_name in local_chemistry_operators():
            problems = InputParser.get_operator_problems(operator_name)
            if problem_name in problems:
                operator_names.append(operator_name)

        return operator_names
