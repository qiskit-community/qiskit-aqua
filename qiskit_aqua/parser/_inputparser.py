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

from qiskit_aqua.algorithmerror import AlgorithmError
import json
from collections import OrderedDict
import logging
import os
import copy
from qiskit_aqua import (local_pluggables_types,
                         get_algorithm_configuration,
                         local_algorithms)
from qiskit_aqua.input import (local_inputs, get_input_configuration)
from .jsonschema import JSONSchema

logger = logging.getLogger(__name__)


class InputParser(object):
    """JSON input Parser."""

    INPUT = 'input'
    _UNKNOWN = 'unknown'
    _PROPERTY_ORDER = [JSONSchema.NAME, _UNKNOWN]

    def __init__(self, input=None):
        """Create InputParser object."""
        self._original_sections = None
        self._filename = None
        self._sections = None
        if input is not None:
            if isinstance(input, dict):
                self._sections = input
            elif isinstance(input, str):
                self._filename = input
            else:
                raise AlgorithmError("Invalid parser input type.")

        self._section_order = [JSONSchema.PROBLEM,
                               InputParser.INPUT, JSONSchema.ALGORITHM]
        for pluggable_type in local_pluggables_types():
            if pluggable_type != JSONSchema.ALGORITHM:
                self._section_order.append(pluggable_type)

        self._section_order.extend([JSONSchema.BACKEND, InputParser._UNKNOWN])

        self._json_schema = JSONSchema(os.path.join(
            os.path.dirname(__file__), 'input_schema.json'))
        self._json_schema.populate_problem_names()
        self._json_schema.commit_changes()

    def _order_sections(self, sections):
        sections_sorted = OrderedDict(sorted(list(sections.items()),
                                             key=lambda x: self._section_order.index(
                                                 x[0])
                                             if x[0] in self._section_order else self._section_order.index(InputParser._UNKNOWN)))

        for section, properties in sections_sorted.items():
            if isinstance(properties, dict):
                sections_sorted[section] = OrderedDict(sorted(list(properties.items()),
                                                              key=lambda x: InputParser._PROPERTY_ORDER.index(
                                                                  x[0])
                                                              if x[0] in InputParser._PROPERTY_ORDER else InputParser._PROPERTY_ORDER.index(InputParser._UNKNOWN)))

        return sections_sorted

    def parse(self):
        """Parse the data."""
        if self._sections is None:
            if self._filename is None:
                raise AlgorithmError("Missing input file")

            with open(self._filename) as json_file:
                self._sections = json.load(json_file)

        self._json_schema.update_pluggable_input_schemas(self)
        self._update_algorithm_input_schema()
        self._sections = self._order_sections(self._sections)
        self._original_sections = copy.deepcopy(self._sections)

    def is_modified(self):
        """
        Returns true if data has been changed
        """
        return self._original_sections != self._sections

    @staticmethod
    def is_pluggable_section(section_name):
        return JSONSchema.format_section_name(section_name) in local_pluggables_types()

    def get_section_types(self, section_name):
        return self._json_schema.get_section_types(section_name)

    def get_property_types(self, section_name, property_name):
        return self._json_schema.get_property_types(section_name, property_name)

    def get_default_sections(self):
        return self._json_schema.get_default_sections()

    def get_default_section_names(self):
        return self._json_schema.get_default_section_names()

    def get_section_default_properties(self, section_name):
        return self._json_schema.get_section_default_properties(section_name)

    def allows_additional_properties(self, section_name):
        return self._json_schema.allows_additional_properties(section_name)

    def get_property_default_values(self, section_name, property_name):
        return self._json_schema.get_property_default_values(section_name, property_name)

    def get_property_default_value(self, section_name, property_name):
        return self._json_schema.get_property_default_value(section_name, property_name)

    def get_filename(self):
        """Return the filename."""
        return self._filename

    @staticmethod
    def get_input_problems(input_name):
        config = get_input_configuration(input_name)
        if 'problems' in config:
            return config['problems']

        return []

    @staticmethod
    def get_algorithm_problems(algo_name):
        return JSONSchema.get_algorithm_problems(algo_name)

    def _update_algorithm_input_schema(self):
        # find alogorithm input
        default_name = self.get_property_default_value(
            InputParser.INPUT, JSONSchema.NAME)
        input_name = self.get_section_property(
            InputParser.INPUT, JSONSchema.NAME, default_name)
        if input_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(
                JSONSchema.PROBLEM, JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(
                    JSONSchema.PROBLEM, JSONSchema.NAME)

            if problem_name is None:
                raise AlgorithmError(
                    "No algorithm 'problem' section found on input.")

            for name in local_inputs():
                if problem_name in self.get_input_problems(name):
                    # set to the first input to solve the problem
                    input_name = name
                    break

        if input_name is None:
            # just remove fromm schema if none solves the problem
            if InputParser.INPUT in self._json_schema.schema['properties']:
                del self._json_schema.schema['properties'][InputParser.INPUT]
            return

        if default_name is None:
            default_name = input_name

        config = {}
        try:
            config = get_input_configuration(input_name)
        except:
            pass

        input_schema = config['input_schema'] if 'input_schema' in config else {
        }
        properties = input_schema['properties'] if 'properties' in input_schema else {
        }
        properties[JSONSchema.NAME] = {'type': 'string'}
        required = input_schema['required'] if 'required' in input_schema else [
        ]
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True
        if default_name is not None:
            properties[JSONSchema.NAME]['default'] = default_name
            required.append(JSONSchema.NAME)

        if InputParser.INPUT not in self._json_schema.schema['properties']:
            self._json_schema.schema['properties'][InputParser.INPUT] = {
                'type': 'object'}

        self._json_schema.schema['properties'][InputParser.INPUT]['properties'] = properties
        self._json_schema.schema['properties'][InputParser.INPUT]['required'] = required
        self._json_schema.schema['properties'][InputParser.INPUT]['additionalProperties'] = additionalProperties

    def _merge_dependencies(self):
        algo_name = self.get_section_property(
            JSONSchema.ALGORITHM, JSONSchema.NAME)
        if algo_name is None:
            return

        config = get_algorithm_configuration(algo_name)
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {
        } if 'defaults' not in config else config['defaults']
        for pluggable_type in local_pluggables_types():
            if pluggable_type != JSONSchema.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from input that are not in the dependencies
                if pluggable_type in self._sections:
                    del self._sections[pluggable_type]

        section_names = self.get_section_names()
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            new_properties = {}
            if pluggable_type in pluggable_defaults:
                for key, value in pluggable_defaults[pluggable_type].items():
                    if key == JSONSchema.NAME:
                        pluggable_name = pluggable_defaults[pluggable_type][key]
                    else:
                        new_properties[key] = value

            if pluggable_name is None:
                continue

            if pluggable_type not in section_names:
                self.set_section(pluggable_type)

            if self.get_section_property(pluggable_type, JSONSchema.NAME) is None:
                self.set_section_property(
                    pluggable_type, JSONSchema.NAME, pluggable_name)

            if pluggable_name == self.get_section_property(pluggable_type, JSONSchema.NAME):
                properties = self.get_section_properties(pluggable_type)
                if new_properties:
                    new_properties.update(properties)
                else:
                    new_properties = properties

                self.set_section_properties(pluggable_type, new_properties)

    def _merge_default_values(self):
        section_names = self.get_section_names()
        if JSONSchema.ALGORITHM in section_names:
            if JSONSchema.PROBLEM not in section_names:
                self.set_section(JSONSchema.PROBLEM)

        self._json_schema.update_pluggable_input_schemas(self)
        self._update_algorithm_input_schema()
        self._merge_dependencies()

        # do not merge any pluggable that doesn't have name default in schema
        default_section_names = []
        pluggable_types = local_pluggables_types()
        for section_name in self.get_default_section_names():
            if section_name in pluggable_types:
                if self.get_property_default_value(section_name, JSONSchema.NAME) is not None:
                    default_section_names.append(section_name)
            else:
                default_section_names.append(section_name)

        section_names = set(self.get_section_names()
                            ) | set(default_section_names)
        for section_name in section_names:
            if section_name not in self._sections:
                self.set_section(section_name)

            new_properties = self.get_section_default_properties(section_name)
            if new_properties is not None:
                if self.section_is_text(section_name):
                    text = self.get_section_text(section_name)
                    if (text is None or len(text) == 0) and \
                            isinstance(new_properties, str) and \
                            len(new_properties) > 0 and \
                            text != new_properties:
                        self.set_section_data(section_name, new_properties)
                else:
                    properties = self.get_section_properties(section_name)
                    new_properties.update(properties)
                    self.set_section_properties(section_name, new_properties)

        self._sections = self._order_sections(self._sections)

    def validate_merge_defaults(self):
        self._merge_default_values()
        self._json_schema.validate(self.get_sections())
        self._validate_algorithm_problem()
        self._validate_input_problem()

    def _validate_algorithm_problem(self):
        algo_name = self.get_section_property(
            JSONSchema.ALGORITHM, JSONSchema.NAME)
        if algo_name is None:
            return

        problem_name = self.get_section_property(
            JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(
                JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AlgorithmError(
                "No algorithm 'problem' section found on input.")

        problems = InputParser.get_algorithm_problems(algo_name)
        if problem_name not in problems:
            raise AlgorithmError(
                "Problem: {} not in the list of problems: {} for algorithm: {}.".format(problem_name, problems, algo_name))

    def _validate_input_problem(self):
        input_name = self.get_section_property(
            InputParser.INPUT, JSONSchema.NAME)
        if input_name is None:
            return

        problem_name = self.get_section_property(
            JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(
                JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AlgorithmError(
                "No algorithm 'problem' section found on input.")

        problems = InputParser.get_input_problems(input_name)
        if problem_name not in problems:
            raise AlgorithmError(
                "Problem: {} not in the list of problems: {} for input: {}.".format(problem_name, problems, input_name))

    def commit_changes(self):
        self._original_sections = copy.deepcopy(self._sections)

    def save_to_file(self, file_name):
        if file_name is None:
            raise AlgorithmError('Missing file path')

        file_name = file_name.strip()
        if len(file_name) == 0:
            raise AlgorithmError('Missing file path')

        with open(file_name, 'w') as f:
            print(json.dumps(self.get_sections(),
                             sort_keys=True, indent=4), file=f)

    def section_is_text(self, section_name):
        section_name = JSONSchema.format_section_name(section_name)
        types = self.get_section_types(section_name)
        if len(types) > 0:
            return 'object' not in types

        section = self.get_section(section_name)
        if section is None:
            return False

        return not isinstance(section, dict)

    def get_sections(self):
        return self._sections

    def get_section(self, section_name):
        """Return a Section by name.
        Args:
            section_name (str): the name of the section, case insensitive
        Returns:
            Section: The section with this name
        Raises:
            AlgorithmError: if the section does not exist.
        """
        section_name = JSONSchema.format_section_name(section_name)
        try:
            return self._sections[section_name]
        except KeyError:
            raise AlgorithmError('No section "{0}"'.format(section_name))

    def get_section_text(self, section_name):
        section = self.get_section(section_name)
        if section is None:
            return ''

        if isinstance(section, str):
            return str

        return json.dumps(section, sort_keys=True, indent=4)

    def get_section_properties(self, section_name):
        section = self.get_section(section_name)
        if section is None:
            return {}

        return section

    def get_section_property(self, section_name, property_name, default_value=None):
        """Return a property by name.
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            default_value : default value in case it is not found
        Returns:
            Value: The property value
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        if section_name in self._sections:
            section = self._sections[section_name]
            if property_name in section:
                return section[property_name]

        return default_value

    def set_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name)
        if section_name not in self._sections:
            self._sections[section_name] = OrderedDict()
            self._sections = self._order_sections(self._sections)

    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name)
        if section_name not in self._sections:
            return

        del self._sections[section_name]

        # update schema
        self._json_schema.rollback_changes()
        self._json_schema.update_pluggable_input_schemas(self)
        self._update_algorithm_input_schema()

    def set_section_properties(self, section_name, properties):
        self.delete_section_properties(section_name)
        for property_name, value in properties.items():
            self.set_section_property(section_name, property_name, value)

    def set_section_property(self, section_name, property_name, value):
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        value = self._json_schema.check_property_value(
            section_name, property_name, value)
        types = self.get_property_types(section_name, property_name)

        sections_temp = copy.deepcopy(self._sections)
        InputParser._set_section_property(
            sections_temp, section_name, property_name, value, types)
        msg = self._json_schema.validate_property(
            sections_temp, section_name, property_name)
        if msg is not None:
            raise AlgorithmError("{}.{}: Value '{}': '{}'".format(
                section_name, property_name, value, msg))

        InputParser._set_section_property(
            self._sections, section_name, property_name, value, types)
        if property_name == JSONSchema.NAME:
            if InputParser.INPUT == section_name:
                self._update_algorithm_input_schema()
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(
                    section_name)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(
                                section_name, property_name)
            elif JSONSchema.PROBLEM == section_name:
                self._update_algorithm_problem()
                self._update_input_problem()
            elif InputParser.is_pluggable_section(section_name):
                self._json_schema.update_pluggable_input_schemas(self)
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(
                    section_name)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(
                                section_name, property_name)

                if section_name == JSONSchema.ALGORITHM:
                    self._update_dependency_sections()

        self._sections = self._order_sections(self._sections)

    def _update_algorithm_problem(self):
        problem_name = self.get_section_property(
            JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(
                JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AlgorithmError(
                "No algorithm 'problem' section found on input.")

        algo_name = self.get_section_property(
            JSONSchema.ALGORITHM, JSONSchema.NAME)
        if algo_name is not None and problem_name in InputParser.get_algorithm_problems(algo_name):
            return

        for algo_name in local_algorithms():
            if problem_name in self.get_algorithm_problems(algo_name):
                # set to the first algorithm to solve the problem
                self.set_section_property(
                    JSONSchema.ALGORITHM, JSONSchema.NAME, algo_name)
                return

        # no algorithm solve this problem, remove section
        self.delete_section(JSONSchema.ALGORITHM)

    def _update_input_problem(self):
        problem_name = self.get_section_property(
            JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(
                JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise AlgorithmError(
                "No algorithm 'problem' section found on input.")

        input_name = self.get_section_property(
            InputParser.INPUT, JSONSchema.NAME)
        if input_name is not None and problem_name in InputParser.get_input_problems(input_name):
            return

        for input_name in local_inputs():
            if problem_name in self.get_input_problems(input_name):
                # set to the first input to solve the problem
                self.set_section_property(
                    InputParser.INPUT, JSONSchema.NAME, input_name)
                return

        # no input solve this problem, remove section
        self.delete_section(InputParser.INPUT)

    def _update_dependency_sections(self):
        algo_name = self.get_section_property(
            JSONSchema.ALGORITHM, JSONSchema.NAME)
        config = {} if algo_name is None else get_algorithm_configuration(
            algo_name)
        classical = config['classical'] if 'classical' in config else False
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {
        } if 'defaults' not in config else config['defaults']
        pluggable_types = local_pluggables_types()
        for pluggable_type in pluggable_types:
            # remove pluggables from input that are not in the dependencies
            if pluggable_type != JSONSchema.ALGORITHM and pluggable_type not in pluggable_dependencies and pluggable_type in self._sections:
                del self._sections[pluggable_type]

        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            if pluggable_type in pluggable_defaults:
                if JSONSchema.NAME in pluggable_defaults[pluggable_type]:
                    pluggable_name = pluggable_defaults[pluggable_type][JSONSchema.NAME]

            if pluggable_name is not None and pluggable_type not in self._sections:
                self.set_section_property(
                    pluggable_type, JSONSchema.NAME, pluggable_name)
                # update default values for new dependency pluggable types
                self.set_section_properties(
                    pluggable_type, self.get_section_default_properties(pluggable_type))

        # update backend based on classical
        if classical:
            if JSONSchema.BACKEND in self._sections:
                del self._sections[JSONSchema.BACKEND]
        else:
            if JSONSchema.BACKEND not in self._sections:
                self._sections[JSONSchema.BACKEND] = self.get_section_default_properties(
                    JSONSchema.BACKEND)

        # reorder sections
        self._sections = self._order_sections(self._sections)

    @staticmethod
    def _set_section_property(sections, section_name, property_name, value, types):
        """
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            value : property value
            types : schema types
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        value = JSONSchema.get_value(value, types)

        if section_name not in sections:
            sections[section_name] = OrderedDict()

        # name should come first
        if JSONSchema.NAME == property_name and property_name not in sections[section_name]:
            new_dict = OrderedDict([(property_name, value)])
            new_dict.update(sections[section_name])
            sections[section_name] = new_dict
        else:
            sections[section_name][property_name] = value

    def delete_section_property(self, section_name, property_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        if section_name in self._sections and property_name in self._sections[section_name]:
            del self._sections[section_name][property_name]

    def delete_section_properties(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name)
        if section_name in self._sections:
            del self._sections[section_name]

    def set_section_data(self, section_name, value):
        """
        Sets a section data.
        Args:
            section_name (str): the name of the section, case insensitive
            value : value to set
        """
        section_name = JSONSchema.format_section_name(section_name)
        self._sections[section_name] = self._json_schema.check_section_value(
            section_name, value)

    def get_section_names(self):
        """Return all the names of the sections."""
        return list(self._sections.keys())
