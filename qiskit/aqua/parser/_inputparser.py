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

"""Aqua input Parser."""

import json
import logging
import os
import copy
from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua import (local_pluggables_types,
                         PluggableType,
                         get_pluggable_configuration,
                         local_pluggables)
from .base_parser import BaseParser
from .json_schema import JSONSchema

logger = logging.getLogger(__name__)


class InputParser(BaseParser):
    """Aqua input Parser."""

    def __init__(self, input_value=None):
        """Create Parser object."""
        super().__init__(JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json')))
        if input is not None:
            if isinstance(input_value, dict):
                self._sections = input_value
            elif isinstance(input_value, str):
                self._filename = input_value
            else:
                raise AquaError("Invalid parser input type.")

        self._section_order = [JSONSchema.PROBLEM,
                               PluggableType.INPUT.value,
                               PluggableType.ALGORITHM.value]
        for pluggable_type in local_pluggables_types():
            if pluggable_type not in [PluggableType.INPUT, PluggableType.ALGORITHM]:
                self._section_order.append(pluggable_type.value)

        self._section_order.extend([JSONSchema.BACKEND, InputParser._UNKNOWN])

    def parse(self):
        """Parse the data."""
        if self._sections is None:
            if self._filename is None:
                raise AquaError("Missing input file")

            with open(self._filename) as json_file:
                self._sections = json.load(json_file)

        self.json_schema.update_backend_schema(self)
        self.json_schema.update_pluggable_schemas(self)
        self._update_algorithm_input_schema()
        self._sections = self._order_sections(self._sections)
        self._original_sections = copy.deepcopy(self._sections)

    def get_default_sections(self):
        return self.json_schema.get_default_sections()

    def merge_default_values(self):
        section_names = self.get_section_names()
        if PluggableType.ALGORITHM.value in section_names:
            if JSONSchema.PROBLEM not in section_names:
                self.set_section(JSONSchema.PROBLEM)

        self.json_schema.update_backend_schema(self)
        self.json_schema.update_pluggable_schemas(self)
        self._update_algorithm_input_schema()
        self._merge_dependencies()

        # do not merge any pluggable that doesn't have name default in schema
        default_section_names = []
        pluggable_type_names = [pluggable_type.value for pluggable_type in local_pluggables_types()]
        for section_name in self.get_default_section_names():
            if section_name in pluggable_type_names:
                if self.get_property_default_value(section_name, JSONSchema.NAME) is not None:
                    default_section_names.append(section_name)
            else:
                default_section_names.append(section_name)

        section_names = set(self.get_section_names()) | set(default_section_names)
        for section_name in section_names:
            if section_name not in self._sections:
                self.set_section(section_name)

            new_properties = self.get_section_default_properties(section_name)
            if new_properties is not None:
                if self.section_is_text(section_name):
                    text = self.get_section_text(section_name)
                    if not text and isinstance(new_properties, str) and new_properties:
                        self.set_section_data(section_name, new_properties)
                else:
                    properties = self.get_section_properties(section_name)
                    new_properties.update(properties)
                    self.set_section_properties(section_name, new_properties)

        self._sections = self._order_sections(self._sections)

    def validate_merge_defaults(self):
        super().validate_merge_defaults()
        self._validate_input_problem()

    def save_to_file(self, file_name):
        if file_name is None:
            raise AquaError('Missing file path')

        file_name = file_name.strip()
        if not file_name:
            raise AquaError('Missing file path')

        with open(file_name, 'w') as file:
            print(json.dumps(self.get_sections(), sort_keys=True, indent=4), file=file)

    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        super().delete_section(section_name)
        self._update_algorithm_input_schema()

    def post_set_section_property(self, section_name, property_name):
        property_name = JSONSchema.format_property_name(property_name)
        if property_name == JSONSchema.NAME:
            section_name = JSONSchema.format_section_name(section_name).lower()
            if PluggableType.INPUT.value == section_name:
                self._update_algorithm_input_schema()
            elif JSONSchema.PROBLEM == section_name:
                self._update_input_problem()
                self._update_algorithm_input_schema()
                # remove properties that are not valid for this input
                default_properties = self.get_section_default_properties(PluggableType.INPUT.value)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(PluggableType.INPUT.value)
                    for p_name in list(properties.keys()):
                        if p_name != JSONSchema.NAME and p_name not in default_properties:
                            self.delete_section_property(PluggableType.INPUT.value, p_name)

    @staticmethod
    def get_input_problems(input_name):
        """ returns input problems """
        config = get_pluggable_configuration(PluggableType.INPUT, input_name)
        if 'problems' in config:
            return config['problems']

        return []

    def _update_algorithm_input_schema(self):
        # find algorithm input
        default_name = self.get_property_default_value(PluggableType.INPUT.value, JSONSchema.NAME)
        input_name = self.get_section_property(PluggableType.INPUT.value,
                                               JSONSchema.NAME, default_name)
        if input_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(JSONSchema.PROBLEM,
                                                               JSONSchema.NAME)

            if problem_name is None:
                raise AquaError("No algorithm 'problem' section found on input.")

            for name in local_pluggables(PluggableType.INPUT):
                if problem_name in self.get_input_problems(name):
                    # set to the first input to solve the problem
                    input_name = name
                    break

        if input_name is None:
            # just remove from schema if none solves the problem
            if PluggableType.INPUT.value in self.json_schema.schema['properties']:
                del self.json_schema.schema['properties'][PluggableType.INPUT.value]
            return

        if default_name is None:
            default_name = input_name

        config = {}
        try:
            config = get_pluggable_configuration(PluggableType.INPUT, input_name)
        except Exception:  # pylint: disable=broad-except
            pass

        input_schema = config['input_schema'] if 'input_schema' in config else {}
        properties = input_schema['properties'] if 'properties' in input_schema else {}
        properties[JSONSchema.NAME] = {'type': 'string'}
        required = input_schema['required'] if 'required' in input_schema else []
        additional_properties = input_schema['additionalProperties'] \
            if 'additionalProperties' in input_schema else True
        if default_name is not None:
            properties[JSONSchema.NAME]['default'] = default_name
            required.append(JSONSchema.NAME)

        if PluggableType.INPUT.value not in self.json_schema.schema['properties']:
            self.json_schema.schema['properties'][PluggableType.INPUT.value] = {'type': 'object'}

        self.json_schema.schema['properties'][PluggableType.INPUT.value]['properties'] = properties
        self.json_schema.schema['properties'][PluggableType.INPUT.value]['required'] = required
        self.json_schema.schema['properties'][PluggableType.INPUT.value]['additionalProperties'] = \
            additional_properties

    def _validate_input_problem(self):
        input_name = self.get_section_property(PluggableType.INPUT.value, JSONSchema.NAME)
        if input_name is None:
            return

        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AquaError("No algorithm 'problem' section found on input.")

        problems = InputParser.get_input_problems(input_name)
        if problem_name not in problems:
            raise AquaError(
                "Problem: {} not in the list of problems: {} for input: {}.".format(
                    problem_name, problems, input_name))

    def _update_input_problem(self):
        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AquaError("No algorithm 'problem' section found on input.")

        input_name = self.get_section_property(PluggableType.INPUT.value, JSONSchema.NAME)
        if input_name is not None and problem_name in InputParser.get_input_problems(input_name):
            return

        for input_name in local_pluggables(PluggableType.INPUT):
            if problem_name in self.get_input_problems(input_name):
                # set to the first input to solve the problem
                self.set_section_property(PluggableType.INPUT.value, JSONSchema.NAME, input_name)
                return

        # no input solve this problem, remove section
        self.delete_section(PluggableType.INPUT.value)
