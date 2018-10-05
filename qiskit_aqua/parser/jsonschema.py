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

import json
import os
import jsonschema
import copy
import ast
from collections import OrderedDict
import logging
from qiskit_aqua import AlgorithmError
from qiskit_aqua import (local_pluggables_types,
                         get_pluggable_configuration,
                         get_algorithm_configuration,
                         local_algorithms)

logger = logging.getLogger(__name__)


class JSONSchema(object):
    """JSON schema Utilities class."""

    NAME = 'name'
    PROBLEM = 'problem'
    ALGORITHM = 'algorithm'
    BACKEND = 'backend'

    def __init__(self, jsonfile):
        """Create JSONSchema object."""
        self._schema = None
        self._original_schema = None
        self.aqua_jsonschema = None
        with open(jsonfile) as json_file:
            self._schema = json.load(json_file)
            validator = jsonschema.Draft4Validator(self._schema)
            self._schema = JSONSchema._resolve_schema_references(
                validator.schema, validator.resolver)
            self.commit_changes()

    @property
    def schema(self):
        """Returns json schema"""
        return self._schema

    @property
    def original_schema(self):
        """Returns original json schema"""
        return self._original_schema

    def commit_changes(self):
        """Saves changes to original json schema"""
        self._original_schema = copy.deepcopy(self._schema)

    def rollback_changes(self):
        """Restores schema from original json schema"""
        self._schema = copy.deepcopy(self._original_schema)

    def populate_problem_names(self):
        """Populate enum list of problem names"""
        problems_dict = OrderedDict()
        for algo_name in local_algorithms():
            problems = JSONSchema.get_algorithm_problems(algo_name)
            for problem in problems:
                problems_dict[problem] = None

        problems_enum = {'enum': list(problems_dict.keys())}
        self._schema['properties'][JSONSchema.PROBLEM]['properties'][JSONSchema.NAME]['oneOf'] = [
            problems_enum]

    def copy_section_from_aqua_schema(self, section_name):
        """
        Copy a section from aqua json schema if if exists
        Args:
            section_name (string): schema section to copy
        """
        section_name = JSONSchema.format_section_name(section_name)
        if self.aqua_jsonschema is None:
            self.aqua_jsonschema = JSONSchema(os.path.join(
                os.path.dirname(__file__), 'input_schema.json'))

        if section_name in self.aqua_jsonschema.schema['properties']:
            self._schema['properties'][section_name] = self.aqua_jsonschema.schema['properties'][section_name]

    def get_section_types(self, section_name):
        """
        Returns types for a schema section

        Args:
            section_name (string): schema section

        Returns:
            Returns schema tyoe array
        """
        section_name = JSONSchema.format_section_name(section_name)
        if 'properties' not in self._schema:
            return []

        if section_name not in self._schema['properties']:
            return []

        if 'type' not in self._schema['properties'][section_name]:
            return []

        types = self._schema['properties'][section_name]['type']
        if isinstance(types, list):
            return types

        return [types]

    def get_property_types(self, section_name, property_name):
        """
        Returns types for a schema section property
        Args:
            section_name (string): schema section
            property_name (string): schema section property

        Returns:
            Returns schema type list
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        if 'properties' not in self._schema:
            return []

        if section_name not in self._schema['properties']:
            return []

        if 'properties' not in self._schema['properties'][section_name]:
            return []

        if property_name not in self._schema['properties'][section_name]['properties']:
            return []

        prop = self._schema['properties'][section_name]['properties'][property_name]
        if 'type' in prop:
            types = prop['type']
            if isinstance(types, list):
                return types

            return [types]

        return []

    def get_default_sections(self):
        """
        Returns default sections
        """
        if 'properties' not in self._schema:
            return None

        return copy.deepcopy(self._schema['properties'])

    def get_default_section_names(self):
        """
        Returns default section names
        """
        sections = self.get_default_sections()
        return list(sections.keys()) if sections is not None else []

    def get_section_default_properties(self, section_name):
        """
        Returns default properties for a schema section

        Args:
            section_name (string): schema section

        Returns:
            Returns properties  dictionary
        """
        section_name = JSONSchema.format_section_name(section_name)
        if 'properties' not in self._schema:
            return None

        if section_name not in self._schema['properties']:
            return None

        types = [self._schema['properties'][section_name]['type']
                 ] if 'type' in self._schema['properties'][section_name] else []

        if 'default' in self._schema['properties'][section_name]:
            return JSONSchema.get_value(self._schema['properties'][section_name]['default'], types)

        if 'object' not in types:
            return JSONSchema.get_value(None, types)

        if 'properties' not in self._schema['properties'][section_name]:
            return None

        properties = OrderedDict()
        for property_name, values in self._schema['properties'][section_name]['properties'].items():
            types = [values['type']] if 'type' in values else []
            default_value = values['default'] if 'default' in values else None
            properties[property_name] = JSONSchema.get_value(
                default_value, types)

        return properties

    def allows_additional_properties(self, section_name):
        """
        Returns allows additional properties flag for a schema section
        Args:
            section_name (string): schema section

        Returns:
            Returns allows additional properties boolean value
        """
        section_name = JSONSchema.format_section_name(section_name)
        if 'properties' not in self._schema:
            return True

        if section_name not in self._schema['properties']:
            return True

        if 'additionalProperties' not in self._schema['properties'][section_name]:
            return True

        return JSONSchema.get_value(self._schema['properties'][section_name]['additionalProperties'])

    def get_property_default_values(self, section_name, property_name):
        """
        Returns default values for a schema section property
        Args:
            section_name (string): schema section
            property_name (string): schema section property

        Returns:
            Returns dafault values list
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        if 'properties' not in self._schema:
            return None

        if section_name not in self._schema['properties']:
            return None

        if 'properties' not in self._schema['properties'][section_name]:
            return None

        if property_name not in self._schema['properties'][section_name]['properties']:
            return None

        prop = self._schema['properties'][section_name]['properties'][property_name]
        if 'type' in prop:
            types = prop['type']
            if not isinstance(types, list):
                types = [types]

            if 'boolean' in types:
                return [True, False]

        if 'oneOf' not in prop:
            return None

        for item in prop['oneOf']:
            if 'enum' in item:
                return item['enum']

        return None

    def get_property_default_value(self, section_name, property_name):
        """
        Returns default value for a schema section property

        Args:
            section_name (string): schema section
            property_name (string): schema section property

        Returns:
            Returns dafault value
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        if 'properties' not in self._schema:
            return None

        if section_name not in self._schema['properties']:
            return None

        if 'properties' not in self._schema['properties'][section_name]:
            return None

        if property_name not in self._schema['properties'][section_name]['properties']:
            return None

        prop = self._schema['properties'][section_name]['properties'][property_name]
        if 'default' in prop:
            return JSONSchema.get_value(prop['default'])

        return None

    def update_pluggable_input_schemas(self, input_parser):
        """
        Updates schemas of all pluggables

        Args:
            input_parser (obj): input parser
        """
        # find alogorithm
        default_algo_name = self.get_property_default_value(
            JSONSchema.ALGORITHM, JSONSchema.NAME)
        algo_name = input_parser.get_section_property(
            JSONSchema.ALGORITHM, JSONSchema.NAME, default_algo_name)

        # update alogorithm scheme
        if algo_name is not None:
            self._update_pluggable_input_schema(
                JSONSchema.ALGORITHM, algo_name, default_algo_name)

        # update alogorithm depoendencies scheme
        config = {} if algo_name is None else get_algorithm_configuration(
            algo_name)
        classical = config['classical'] if 'classical' in config else False
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {
        } if 'defaults' not in config else config['defaults']
        pluggable_types = local_pluggables_types()
        for pluggable_type in pluggable_types:
            if pluggable_type != JSONSchema.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from schema that ate not in the dependencies
                if pluggable_type in self._schema['properties']:
                    del self._schema['properties'][pluggable_type]

        # update algorithm backend from schema if it is classical or not
        if classical:
            if JSONSchema.BACKEND in self._schema['properties']:
                del self._schema['properties'][JSONSchema.BACKEND]
        else:
            if JSONSchema.BACKEND not in self._schema['properties']:
                self._schema['properties'][JSONSchema.BACKEND] = self._original_schema['properties'][JSONSchema.BACKEND]

        # update schema with dependencies
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            default_properties = {}
            if pluggable_type in pluggable_defaults:
                for key, value in pluggable_defaults[pluggable_type].items():
                    if key == JSONSchema.NAME:
                        pluggable_name = pluggable_defaults[pluggable_type][key]
                    else:
                        default_properties[key] = value

            default_name = pluggable_name
            pluggable_name = input_parser.get_section_property(
                pluggable_type, JSONSchema.NAME, pluggable_name)

            # update dependency schema
            self._update_pluggable_input_schema(
                pluggable_type, pluggable_name, default_name)
            for property_name in self._schema['properties'][pluggable_type]['properties'].keys():
                if property_name in default_properties:
                    self._schema['properties'][pluggable_type]['properties'][property_name]['default'] = default_properties[property_name]

    def _update_pluggable_input_schema(self, pluggable_type, pluggable_name, default_name):
        config = {}
        try:
            if pluggable_type is not None and pluggable_name is not None:
                config = get_pluggable_configuration(
                    pluggable_type, pluggable_name)
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

        if pluggable_type not in self._schema['properties']:
            self._schema['properties'][pluggable_type] = {'type': 'object'}

        self._schema['properties'][pluggable_type]['properties'] = properties
        if len(required) > 0:
            self._schema['properties'][pluggable_type]['required'] = required
        elif 'required' in self._schema['properties'][pluggable_type]:
            del self._schema['properties'][pluggable_type]['required']

        self._schema['properties'][pluggable_type]['additionalProperties'] = additionalProperties

    def check_section_value(self, section_name, value):
        """
        Check value for section name

        Args:
            section_name (string): section name
            value (obj): value

        Returns:
            Returns converted value if valid
        """
        section_name = JSONSchema.format_section_name(section_name)
        types = self.get_section_types(section_name)
        value = JSONSchema.get_value(value, types)
        if len(types) > 0:
            validator = jsonschema.Draft4Validator(self._schema)
            valid = False
            for type in types:
                valid = validator.is_type(value, type)
                if valid:
                    break

            if not valid:
                raise AlgorithmError("{}: Value '{}' is not of types: '{}'".format(
                    section_name, value, types))

        return value

    def check_property_value(self, section_name, property_name, value):
        """
        Check value for property name

        Args:
            section_name (string): section name
            property_name (string): property name
            value (obj): value

        Returns:
            Returns converted value if valid
        """
        section_name = JSONSchema.format_section_name(section_name)
        property_name = JSONSchema.format_property_name(property_name)
        types = self.get_property_types(section_name, property_name)
        value = JSONSchema.get_value(value, types)
        if len(types) > 0:
            validator = jsonschema.Draft4Validator(self._schema)
            valid = False
            for type in types:
                valid = validator.is_type(value, type)
                if valid:
                    break

            if not valid:
                raise AlgorithmError("{}.{} Value '{}' is not of types: '{}'".format(
                    section_name, property_name, value, types))

        return value

    def validate(self, sections_json):
        try:
            logger.debug('JSON Input: {}'.format(
                json.dumps(sections_json, sort_keys=True, indent=4)))
            logger.debug('Aqua Input Schema: {}'.format(
                json.dumps(self._schema, sort_keys=True, indent=4)))
            jsonschema.validate(sections_json, self._schema)
        except jsonschema.exceptions.ValidationError as ve:
            logger.info('JSON Validation error: {}'.format(str(ve)))
            raise AlgorithmError(ve.message)

    def validate_property(self, sections_json, section_name, property_name):
        """
        Validates the propery and returns error message
        Args:
            sections_json(dict): sesctions
            section_name (string): section name
            property_name (string): property name

        Returns:
            Returns error meessage or None
        """
        validator = jsonschema.Draft4Validator(self._schema)
        for error in sorted(validator.iter_errors(sections_json), key=str):
            if len(error.path) == 2 and error.path[0] == section_name and error.path[1] == property_name:
                return error.message

        return None

    @staticmethod
    def get_algorithm_problems(algo_name):
        """
        Get algorithm problem name list
        Args:
            algo_name (string): algorithm name

        Returns:
            Returns list of problem names
        """
        config = get_algorithm_configuration(algo_name)
        if 'problems' in config:
            return config['problems']

        return []

    @staticmethod
    def get_value(value, types=[]):
        """
        Returns a converted value based on schema types
        Args:
            value (obj): value
            type (array): schema types

        Returns:
            Returns converted value
        """
        if value is None or (isinstance(value, str) and len(value.strip()) == 0):
            # return propet values based on type
            if value is None:
                if 'null' in types:
                    return None
                if 'string' in types:
                    return ''
            else:
                if 'string' in types:
                    return value
                if 'null' in types:
                    return None

            if 'integer' in types or 'number' in types:
                return 0
            if 'object' in types:
                return {}
            if 'array' in types:
                return []
            if 'boolean' in types:
                return False

            return value

        if 'number' in types or 'integer' in types:
            try:
                if 'integer' in types:
                    return int(value)
                else:
                    return float(value)
            except ValueError:
                return 0

        if 'string' in types:
            return str(value)

        try:
            str_value = str(value).strip().replace('\n', '').replace('\r', '')
            if str_value.lower() == 'true':
                return True
            elif str_value.lower() == 'false':
                return False

            v = ast.literal_eval(str_value)
            if isinstance(v, dict):
                v = json.loads(json.dumps(v))

            return v
        except:
            return value

    @staticmethod
    def format_section_name(section_name):
        if section_name is None:
            section_name = ''
        section_name = section_name.strip()
        if len(section_name) == 0:
            raise AlgorithmError("Empty section name.")

        return section_name

    @staticmethod
    def format_property_name(property_name):
        if property_name is None:
            property_name = ''
        property_name = property_name.strip()
        if len(property_name) == 0:
            raise AlgorithmError("Empty property name.")

        return property_name

    @staticmethod
    def _resolve_schema_references(schema, resolver):
        """
        Resolves json references and merges them into the schema
        Args:
            schema (dict): schema
            resolver (ob): Validator Resolver

        Returns:
            Returns schema merged with resolved references
        """
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == '$ref':
                    ref_schema = resolver.resolve(value)
                    if ref_schema:
                        return ref_schema[1]

                resolved_ref = JSONSchema._resolve_schema_references(
                    value, resolver)
                if resolved_ref:
                    schema[key] = resolved_ref

        elif isinstance(schema, list):
            for (idx, value) in enumerate(schema):
                resolved_ref = JSONSchema._resolve_schema_references(
                    value, resolver)
                if resolved_ref:
                    schema[idx] = resolved_ref

        return schema
