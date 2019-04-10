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

from qiskit.aqua.parser import BaseParser
import json
from collections import OrderedDict
import logging
import os
import copy
from qiskit.aqua import local_pluggables_types, PluggableType
import pprint
import ast
from qiskit.chemistry import QiskitChemistryError, ChemistryProblem
from qiskit.aqua.parser import JSONSchema
from qiskit.chemistry.core import local_chemistry_operators, get_chemistry_operator_configuration
from qiskit.chemistry.drivers import local_drivers, get_driver_configuration

logger = logging.getLogger(__name__)


class InputParser(BaseParser):
    """Chemistry input file parser."""

    OPERATOR = 'operator'
    DRIVER = 'driver'
    AUTO_SUBSTITUTIONS = 'auto_substitutions'
    _OLD_ENABLE_SUBSTITUTIONS = 'enable_substitutions'

    _START_COMMENTS = ['#', '%']
    _START_SECTION = '&'
    _END_SECTION = '&end'
    _PROPVALUE_SEPARATOR = '='

    _OPTIMIZER = 'optimizer'
    _VARIATIONAL_FORM = 'variational_form'
    _HDF5_INPUT = 'hdf5_input'
    HDF5_OUTPUT = 'hdf5_output'
    _DRIVER_NAMES = None

    def __init__(self, input=None):
        """Create Parser object."""
        json_schema = JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json'))

        # get some properties from algorithms schema
        json_schema.copy_section_from_aqua_schema(PluggableType.ALGORITHM.value)
        json_schema.copy_section_from_aqua_schema(JSONSchema.BACKEND)
        json_schema.copy_section_from_aqua_schema(JSONSchema.PROBLEM)
        json_schema.schema['properties'][JSONSchema.PROBLEM]['properties'][InputParser.AUTO_SUBSTITUTIONS] = {
            "type": "boolean",
            "default": "true"
        }
        super().__init__(json_schema)

        # limit Chemistry problems to energy and excited_states
        chemistry_problems = [problem for problem in
                              self.json_schema.get_property_default_values(JSONSchema.PROBLEM, JSONSchema.NAME)
                              if any(problem == item.value for item in ChemistryProblem)]
        self.json_schema.schema['properties'][JSONSchema.PROBLEM]['properties'][JSONSchema.NAME]['oneOf'] = \
            [{'enum': chemistry_problems}]
        self._json_schema.commit_changes()
        # ---

        self._inputdict = None
        if input is not None:
            if isinstance(input, dict):
                self._inputdict = input
            elif isinstance(input, str):
                self._filename = input
            else:
                raise QiskitChemistryError("Invalid parser input type.")

        self._section_order = [JSONSchema.NAME,
                               JSONSchema.PROBLEM,
                               InputParser.DRIVER,
                               InputParser._UNKNOWN,
                               InputParser.OPERATOR,
                               PluggableType.ALGORITHM.value]
        for pluggable_type in local_pluggables_types():
            if pluggable_type not in [PluggableType.INPUT, PluggableType.ALGORITHM]:
                self._section_order.append(pluggable_type.value)

        self._section_order.extend([JSONSchema.BACKEND, InputParser._UNKNOWN])

        jsonfile = os.path.join(os.path.dirname(__file__), 'substitutions.json')
        with open(jsonfile) as json_file:
            self._substitutions = json.load(json_file)

    def parse(self):
        """Parse the data."""
        if self._inputdict is None:
            if self._filename is None:
                raise QiskitChemistryError("Missing input file")

            section = None
            self._sections = OrderedDict()
            contents = ''
            with open(self._filename, 'rt', encoding="utf8", errors='ignore') as f:
                for line in f:
                    contents += line
                    section = self._process_line(section, line)

            if self._sections:
                # convert to aqua compatible json dictionary based on schema
                driver_configs = OrderedDict()
                for driver_name in local_drivers():
                    driver_configs[driver_name.lower()] = get_driver_configuration(driver_name)

                json_dict = OrderedDict()
                for section_name, section in self._sections.items():
                    types = []
                    if section_name.lower() in driver_configs:
                        config = driver_configs[section_name.lower()]
                        input_schema = copy.deepcopy(config['input_schema']) if 'input_schema' in config else {'type': 'string'}
                        if 'type' not in input_schema:
                            input_schema['type'] = 'string'

                        types = [input_schema['type']]
                    else:
                        types = self.get_section_types(section_name.lower())

                    if 'string' in types:
                        json_dict[section_name] = section['data'] if 'data' in section else ''
                    else:
                        json_dict[section_name] = section['properties'] if 'properties' in section else OrderedDict()

                self._sections = json_dict
            else:
                contents = contents.strip().replace('\n', '').replace('\r', '')
                if len(contents) > 0:
                    # check if input file was dictionary
                    try:
                        v = ast.literal_eval(contents)
                        if isinstance(v, dict):
                            self._inputdict = json.loads(json.dumps(v))
                            self._load_parser_from_dict()
                    except:
                        pass
        else:
            self._load_parser_from_dict()

        # check for old enable_substitutions name
        old_enable_substitutions = self.get_section_property(JSONSchema.PROBLEM, InputParser._OLD_ENABLE_SUBSTITUTIONS)
        if old_enable_substitutions is not None:
            self.delete_section_property(JSONSchema.PROBLEM, InputParser._OLD_ENABLE_SUBSTITUTIONS)
            self.set_section_property(JSONSchema.PROBLEM, InputParser.AUTO_SUBSTITUTIONS, old_enable_substitutions)

        self.json_schema.update_backend_schema(self)
        self.json_schema.update_pluggable_schemas(self)
        self._update_driver_input_schemas()
        self._update_operator_input_schema()
        self._sections = self._order_sections(self._sections)
        self._original_sections = copy.deepcopy(self._sections)

    def get_default_sections(self):
        properties = self.json_schema.get_default_sections()
        driver_name = self.get_section_property(InputParser.DRIVER, JSONSchema.NAME)
        if driver_name is not None:
            properties[driver_name.lower()] = {
                "type": "object"
            }
        return properties

    def merge_default_values(self):
        section_names = self.get_section_names()
        if JSONSchema.NAME not in section_names:
            self.set_section(JSONSchema.NAME)

        if PluggableType.ALGORITHM.value in section_names:
            if JSONSchema.PROBLEM not in section_names:
                self.set_section(JSONSchema.PROBLEM)

        self.json_schema.update_backend_schema(self)
        self.json_schema.update_pluggable_schemas(self)
        self._merge_dependencies()
        self._update_driver_sections()
        self._update_driver_input_schemas()
        self._update_operator_input_schema()

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
        super().validate_merge_defaults()
        self._validate_operator_problem()

    def save_to_file(self, file_name):
        if file_name is None:
            raise QiskitChemistryError('Missing file path')

        file_name = file_name.strip()
        if len(file_name) == 0:
            raise QiskitChemistryError('Missing file path')

        prev_filename = self.get_filename()
        sections = copy.deepcopy(self.get_sections())
        if prev_filename is not None:
            prev_dirname = os.path.dirname(os.path.realpath(prev_filename))
            dirname = os.path.dirname(os.path.realpath(file_name))
            if prev_dirname != dirname:
                InputParser._from_relative_to_abs_paths(sections, prev_filename)

        contents = ''
        lastIndex = len(sections) - 1
        for i, (section_name, section) in enumerate(sections.items()):
            contents += '{}{}'.format(InputParser._START_SECTION, section_name)
            if self.section_is_text(section_name):
                value = section if isinstance(section, str) else json.dumps(section, sort_keys=True, indent=4)
                contents += '\n{}'.format(value)
            else:
                for k, v in section.items():
                    contents += '\n   {}{}{}'.format(k, InputParser._PROPVALUE_SEPARATOR, str(v))

            contents += '\n{}'.format(InputParser._END_SECTION)
            if i < lastIndex:
                contents += '\n\n'

        with open(file_name, 'w') as f:
            print(contents, file=f)

    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        super().delete_section(section_name)
        self._update_driver_input_schemas()
        self._update_operator_input_schema()

    def post_set_section_property(self, section_name, property_name):
        property_name = JSONSchema.format_property_name(property_name)
        if property_name == JSONSchema.NAME:
            section_name = JSONSchema.format_section_name(section_name).lower()
            value = self.get_section_property(section_name, property_name)
            if InputParser.OPERATOR == section_name:
                self._update_operator_input_schema()
            elif JSONSchema.PROBLEM == section_name:
                self._update_operator_problem()
                self._update_operator_input_schema()
                # remove properties that are not valid for this operator
                default_properties = self.get_section_default_properties(InputParser.OPERATOR)
                if isinstance(default_properties, dict):
                    properties = self.get_section_properties(InputParser.OPERATOR)
                    for p_name in list(properties.keys()):
                        if p_name != JSONSchema.NAME and p_name not in default_properties:
                            self.delete_section_property(InputParser.OPERATOR, p_name)
            elif value is not None:
                value = str(value).lower().strip()
                if len(value) > 0 and self.section_is_driver(value):
                    self._update_driver_input_schemas()
                    self._update_driver_sections()

    def is_substitution_allowed(self):
        auto_substitutions = self.get_property_default_value(JSONSchema.PROBLEM, InputParser.AUTO_SUBSTITUTIONS)
        auto_substitutions = self.get_section_property(JSONSchema.PROBLEM, InputParser.AUTO_SUBSTITUTIONS, auto_substitutions)
        if auto_substitutions is None:
            auto_substitutions = True

        return auto_substitutions

    def check_if_substitution_key(self, section_name, property_names):
        result = [(property_name, False) for property_name in property_names]
        if not self.is_substitution_allowed():
            return result

        section_name = JSONSchema.format_section_name(section_name).lower()
        property_names = [JSONSchema.format_property_name(property_name) for property_name in property_names]
        section_property_name = self.get_property_default_value(section_name, JSONSchema.NAME)
        section_property_name = self.get_section_property(section_name, JSONSchema.NAME, section_property_name)
        for key in self._substitutions.keys():
            key_items = key.split('.')
            if len(key_items) == 3 and \
                    key_items[0] == section_name and \
                    key_items[1] == section_property_name and \
                    key_items[2] in property_names:
                result[property_names.index(key_items[2])] = (
                    key_items[2], True)
                continue

        return result

    def process_substitutions(self, substitutions=None):
        if substitutions is not None and not isinstance(substitutions, dict):
            raise QiskitChemistryError(
                'Invalid substitution parameter: {}'.format(substitutions))

        if not self.is_substitution_allowed():
            return {}

        result = {}
        for key, value in self._substitutions.items():
            key_items = key.split('.')
            if len(key_items) != 3:
                raise QiskitChemistryError('Invalid substitution key: {}'.format(key))

            name = self.get_property_default_value(key_items[0], JSONSchema.NAME)
            name = self.get_section_property(key_items[0], JSONSchema.NAME, name)
            if name != key_items[1]:
                continue

            value_set = False
            value_items = value.split('.')
            if len(value_items) == 3:
                name = self.get_section_property(value_items[0], JSONSchema.NAME)
                if name == value_items[1]:
                    v = self.get_property_default_value(value_items[0], value_items[2])
                    v = self.get_section_property(value_items[0], value_items[2], v)
                    if v is not None:
                        self.set_section_property(key_items[0], key_items[2], v)
                        result[key] = v
                        value_set = True

            if value_set or substitutions is None:
                continue

            if value in substitutions:
                self.set_section_property(key_items[0], key_items[2], substitutions[value])
                result[key] = substitutions[value]

        return result

    def _process_line(self, section, line):
        stripLine = line.strip()
        if len(stripLine) == 0:
            if section is not None:
                section['data'].append(line)

            return section

        if stripLine.lower().startswith(InputParser._END_SECTION):
            if section is not None:
                self._sections[section[JSONSchema.NAME]] = self._process_section(section)
            return None

        if stripLine.startswith(InputParser._START_SECTION):
            if section is not None:
                raise QiskitChemistryError('New section "{0}" starting before the end of previuos section "{1}"'.format(
                    line, section[JSONSchema.NAME]))

            return OrderedDict([(JSONSchema.NAME, stripLine[1:].lower()), ('data', [])])

        if section is None:
            return section

        section['data'].append(line)

        return section

    def _process_section(self, section):
        contents = ''
        sep_pos = -len(os.linesep)
        lastIndex = len(section['data']) - 1
        for i, line in enumerate(section['data']):
            key, value = self._get_key_value(line)
            if key is not None and value is not None:
                if 'properties' not in section:
                    section['properties'] = OrderedDict()

                section['properties'][key] = value

            if i == lastIndex:
                if len(line) >= len(os.linesep) and line[sep_pos:] == os.linesep:
                    line = line[:sep_pos]

            contents += line

        section['data'] = contents
        return section

    @staticmethod
    def _get_key_value(line):
        stripLine = line.strip()
        pos = -1
        for start_comment in InputParser._START_COMMENTS:
            pos = stripLine.find(start_comment)
            if pos >= 0:
                break

        if pos == 0:
            return (None, None)

        if pos > 0:
            stripLine = stripLine[:pos].strip()

        pos = stripLine.find(InputParser._PROPVALUE_SEPARATOR)
        if pos > 0:
            key = stripLine[0:pos].strip()
            value = stripLine[pos + 1:].strip()
            return (key, JSONSchema.get_value(value))

        return (None, None)

    @staticmethod
    def get_operator_problems(input_name):
        config = get_chemistry_operator_configuration(input_name)
        if 'problems' in config:
            return config['problems']

        return []

    def _load_parser_from_dict(self):
        self._sections = OrderedDict()
        for section_name, value in self._inputdict.items():
            section_name = JSONSchema.format_section_name(section_name).lower()
            if isinstance(value, dict):
                self._sections[section_name] = OrderedDict(value)
            elif isinstance(value, list) or isinstance(value, str):
                if isinstance(value, list):
                    self._sections[section_name] = '\n'.join(str(e) for e in value)
                else:
                    self._sections[section_name] = value
            else:
                raise QiskitChemistryError("Invalid parser input type for section {}".format(section_name))

    def _update_operator_input_schema(self):
        # find operator
        default_name = self.get_property_default_value(InputParser.OPERATOR, JSONSchema.NAME)
        operator_name = self.get_section_property(InputParser.OPERATOR, JSONSchema.NAME, default_name)
        if operator_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

            if problem_name is None:
                raise QiskitChemistryError("No algorithm 'problem' section found on input.")

            for name in local_chemistry_operators():
                if problem_name in self.get_operator_problems(name):
                    # set to the first input to solve the problem
                    operator_name = name
                    break

        if operator_name is None:
            # just remove fromm schema if none solves the problem
            if InputParser.OPERATOR in self.json_schema.schema['properties']:
                del self.json_schema.schema['properties'][InputParser.OPERATOR]

            return

        if default_name is None:
            default_name = operator_name

        config = {}
        try:
            config = get_chemistry_operator_configuration(operator_name)
        except:
            pass

        input_schema = config['input_schema'] if 'input_schema' in config else {}
        properties = input_schema['properties'] if 'properties' in input_schema else {}
        properties[JSONSchema.NAME] = {'type': 'string'}
        required = input_schema['required'] if 'required' in input_schema else []
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True
        if default_name is not None:
            properties[JSONSchema.NAME]['default'] = default_name
            required.append(JSONSchema.NAME)

        if InputParser.OPERATOR not in self.json_schema.schema['properties']:
            self.json_schema.schema['properties'][InputParser.OPERATOR] = {'type': 'object'}

        self.json_schema.schema['properties'][InputParser.OPERATOR]['properties'] = properties
        self.json_schema.schema['properties'][InputParser.OPERATOR]['required'] = required
        self.json_schema.schema['properties'][InputParser.OPERATOR]['additionalProperties'] = additionalProperties

    def _update_driver_input_schemas(self):
        # find driver name
        default_name = self.get_property_default_value(InputParser.DRIVER, JSONSchema.NAME)
        driver_name = self.get_section_property(InputParser.DRIVER, JSONSchema.NAME, default_name)
        if driver_name is not None:
            driver_name = driver_name.strip().lower()

        for name in local_drivers():
            name_orig = name
            name = name.lower()
            if driver_name is not None and driver_name == name:
                config = get_driver_configuration(name_orig)
                input_schema = copy.deepcopy(config['input_schema']) if 'input_schema' in config else {'type': 'object'}
                if '$schema' in input_schema:
                    del input_schema['$schema']
                if 'id' in input_schema:
                    del input_schema['id']

                self.json_schema.schema['properties'][driver_name] = input_schema
            else:
                if name in self.json_schema.schema['properties']:
                    del self.json_schema.schema['properties'][name]

    @staticmethod
    def _load_driver_names():
        if InputParser._DRIVER_NAMES is None:
            InputParser._DRIVER_NAMES = [name.lower() for name in local_drivers()]

    def _validate_operator_problem(self):
        operator_name = self.get_section_property(InputParser.OPERATOR, JSONSchema.NAME)
        if operator_name is None:
            return

        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise QiskitChemistryError("No algorithm 'problem' section found on input.")

        problems = InputParser.get_operator_problems(operator_name)
        if problem_name not in problems:
            raise QiskitChemistryError("Problem: {} not in the list of problems: {} for operator: {}.".format(problem_name, problems, operator_name))

    def to_dictionary(self):
        dict = OrderedDict()
        for section_name in self.get_section_names():
            if self.section_is_text(section_name):
                dict[section_name] = self.get_section_text(section_name).splitlines()
            else:
                dict[section_name] = self.get_section_properties(section_name)

        return dict

    def export_dictionary(self, file_name):
        if file_name is None:
            raise QiskitChemistryError('Missing file path')

        file_name = file_name.strip()
        if len(file_name) == 0:
            raise QiskitChemistryError('Missing file path')

        value = json.loads(json.dumps(self.to_dictionary()))
        value = pprint.pformat(value, indent=4)
        with open(file_name, 'w') as f:
            print(value, file=f)

    @staticmethod
    def _from_relative_to_abs_paths(sections, filename):
        directory = os.path.dirname(filename)
        for section_name, section in sections.items():
            if isinstance(section, dict):
                for key, value in section.items():
                    if key == InputParser._HDF5_INPUT:
                        if value is not None and not os.path.isabs(value):
                            value = os.path.abspath(os.path.join(directory, value))
                            InputParser._set_section_property(sections, section_name, key, value, ['string'])

    def section_is_driver(self, section_name):
        section_name = JSONSchema.format_section_name(section_name).lower()
        InputParser._load_driver_names()
        return section_name in InputParser._DRIVER_NAMES

    def _update_operator_problem(self):
        problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

        if problem_name is None:
            raise QiskitChemistryError("No algorithm 'problem' section found on input.")

        operator_name = self.get_section_property(InputParser.OPERATOR, JSONSchema.NAME)
        if operator_name is not None and problem_name in InputParser.get_operator_problems(operator_name):
            return

        for operator_name in local_chemistry_operators():
            if problem_name in self.get_operator_problems(operator_name):
                # set to the first input to solve the problem
                self.set_section_property(InputParser.OPERATOR, JSONSchema.NAME, operator_name)
                return

        # no input solve this problem, remove section
        self.delete_section(InputParser.OPERATOR)

    def _update_driver_sections(self):
        driver_name = self.get_section_property(InputParser.DRIVER, JSONSchema.NAME)
        if driver_name is not None:
            driver_name = driver_name.strip().lower()

        for name in local_drivers():
            name = name.lower()
            if driver_name is not None and driver_name == name:
                continue

            if name in self._sections:
                del self._sections[name]

        if driver_name is not None and driver_name not in self._sections:
            self.set_section(driver_name)
            value = self.get_section_default_properties(driver_name)
            if isinstance(value, dict):
                for property_name, property_value in value.items():
                    self.set_section_property(driver_name, property_name, property_value)
            else:
                if value is None:
                    types = self.get_section_types(driver_name)
                    if 'null' not in types:
                        if 'string' in types:
                            value = ''
                        elif 'object' in types:
                            value = {}
                        elif 'array' in types:
                            value = []

                self.set_section_data(driver_name, value)
