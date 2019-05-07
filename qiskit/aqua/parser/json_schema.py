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

import json
import os
import jsonschema
import copy
import ast
from collections import OrderedDict
import logging
from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua import (local_pluggables_types,
                         PluggableType,
                         get_pluggable_configuration,
                         local_pluggables,
                         get_local_providers)
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_simulator_backend,
                                             has_ibmq,
                                             get_backend_from_provider,
                                             get_backends_from_provider,
                                             is_local_backend,
                                             is_aer_provider,
                                             is_aer_statevector_backend)

logger = logging.getLogger(__name__)


class JSONSchema(object):
    """JSON schema Utilities class."""

    NAME = 'name'
    PROVIDER = 'provider'
    PROBLEM = 'problem'
    BACKEND = 'backend'

    def __init__(self, schema_input):
        """Create JSONSchema object."""
        self._schema = None
        self._original_schema = None
        self.aqua_jsonschema = None
        if isinstance(schema_input, dict):
            self._schema = copy.deepcopy(schema_input)
        elif isinstance(schema_input, str):
            with open(schema_input) as json_file:
                self._schema = json.load(json_file)
        else:
            raise AquaError("Invalid JSONSchema input type.")

        validator = jsonschema.Draft4Validator(self._schema)
        self._schema = JSONSchema._resolve_schema_references(validator.schema, validator.resolver)
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

    def _initialize_problem_section(self):
        """Initialize problem"""
        self._schema['properties'][JSONSchema.PROBLEM]['properties']['num_processes']['maximum'] = aqua_globals.CPU_COUNT
        problems_dict = OrderedDict()
        for algo_name in local_pluggables(PluggableType.ALGORITHM):
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
            self.aqua_jsonschema = JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json'))

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

        types = [self._schema['properties'][section_name]['type']] if 'type' in self._schema['properties'][section_name] else []

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
            properties[property_name] = JSONSchema.get_value(default_value, types)

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

    def update_backend_schema(self, input_parser):
        """
        Updates backend schema
        """
        if JSONSchema.BACKEND not in self._schema['properties']:
            return

        # Updates defaults provider/backend
        default_provider_name = None
        default_backend_name = None
        orig_backend_properties = self._original_schema.get('properties', {}).get(JSONSchema.BACKEND, {}).get('properties')
        if orig_backend_properties is not None:
            default_provider_name = orig_backend_properties.get(JSONSchema.PROVIDER, {}).get('default')
            default_backend_name = orig_backend_properties.get(JSONSchema.NAME, {}).get('default')

        providers = get_local_providers()
        if default_provider_name is None or default_provider_name not in providers:
            # use first provider available
            providers_items = providers.items()
            provider_tuple = next(iter(providers_items)) if len(providers_items) > 0 else ('', [])
            default_provider_name = provider_tuple[0]

        if default_backend_name is None or default_backend_name not in providers.get(default_provider_name, []):
            # use first backend available in provider
            default_backend_name = providers.get(default_provider_name)[0] if len(providers.get(default_provider_name, [])) > 0 else ''

        self._schema['properties'][JSONSchema.BACKEND] = {
            'type': 'object',
            'properties': {
                JSONSchema.PROVIDER: {
                    'type': 'string',
                    'default': default_provider_name
                },
                JSONSchema.NAME: {
                    'type': 'string',
                    'default': default_backend_name
                },
            },
            'required': [JSONSchema.PROVIDER, JSONSchema.NAME],
            'additionalProperties': False,
        }
        provider_name = input_parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER, default_provider_name)
        backend_names = get_backends_from_provider(provider_name)
        backend_name = input_parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME, default_backend_name)
        if backend_name not in backend_names:
            # use first backend available in provider
            backend_name = backend_names[0] if len(backend_names) > 0 else ''

        backend = get_backend_from_provider(provider_name, backend_name)
        config = backend.configuration()

        # Include shots in schema only if not a statevector backend.
        # For statevector, shots will be set to 1, in QiskitAqua
        if not is_statevector_backend(backend):
            self._schema['properties'][JSONSchema.BACKEND]['properties']['shots'] = {
                'type': 'integer',
                'minimum': 1,
            }
            default_shots = 1024
            # ensure default_shots <= max_shots
            if config.max_shots:
                default_shots = min(default_shots, config.max_shots)
                self._schema['properties'][JSONSchema.BACKEND]['properties']['shots']['maximum'] = config.max_shots

            self._schema['properties'][JSONSchema.BACKEND]['properties']['shots']['default'] = default_shots

        self._schema['properties'][JSONSchema.BACKEND]['properties']['skip_transpiler'] = {
            'type': 'boolean',
            'default': False,
        }

        coupling_map_devices = []
        noise_model_devices = []
        check_coupling_map = is_simulator_backend(backend)
        check_noise_model = is_aer_provider(backend) and not is_aer_statevector_backend(backend)
        try:
            if (check_coupling_map or check_noise_model) and has_ibmq():
                backend_names = get_backends_from_provider('qiskit.IBMQ')
                for backend_name in backend_names:
                    ibmq_backend = get_backend_from_provider('qiskit.IBMQ', backend_name)
                    if is_simulator_backend(ibmq_backend):
                        continue
                    if check_noise_model:
                        noise_model_devices.append('qiskit.IBMQ:' + backend_name)
                    if check_coupling_map and ibmq_backend.configuration().coupling_map:
                        coupling_map_devices.append('qiskit.IBMQ:' + backend_name)
        except Exception as e:
            logger.debug("Failed to load IBMQ backends. Error {}".format(str(e)))

        # Includes 'coupling map' and 'coupling_map_from_device' in schema only if a simulator backend.
        # Actual devices have a coupling map based on the physical configuration of the device.
        # The user can configure the coupling map so its the same as the coupling map
        # of a given device in order to better simulate running on the device.
        # Property 'coupling_map_from_device' is a list of provider:name backends that are
        # real devices e.g qiskit.IBMQ:ibmqx5.
        # If property 'coupling_map', an array, is provided, it overrides coupling_map_from_device,
        # the latter defaults to 'None'. So in total no coupling map is a default, i.e. all to all coupling is possible.
        if is_simulator_backend(backend):
            self._schema['properties'][JSONSchema.BACKEND]['properties']['coupling_map'] = {
                'type': ['array', 'null'],
                'default': None,
            }
            if len(coupling_map_devices) > 0:
                coupling_map_devices.append(None)
                self._schema['properties'][JSONSchema.BACKEND]['properties']['coupling_map_from_device'] = {
                    'type': ['string', 'null'],
                    'default': None,
                    'oneOf': [
                        {
                            'enum': coupling_map_devices
                        }
                    ],
                }

        # noise model that can be setup for Aer simulator so as to model noise of an actual device.
        if len(noise_model_devices) > 0:
            noise_model_devices.append(None)
            self._schema['properties'][JSONSchema.BACKEND]['properties']['noise_model'] = {
                'type': ['string', 'null'],
                'default': None,
                'oneOf': [
                    {
                        'enum': noise_model_devices
                    }
                ],
            }

        # If a noise model is supplied then the basis gates is set as per the noise model
        # unless basis gates is not None in which case it overrides noise model and a warning msg is logged.
        # as it is an advanced use case.
        self._schema['properties'][JSONSchema.BACKEND]['properties']['basis_gates'] = {
            'type': ['array', 'null'],
            'default': None,
        }

        # TODO: Not sure if we want to continue with initial_layout in declarative form.
        # It requires knowledge of circuit registers etc. Perhaps its best to leave this detail to programming API.
        self._schema['properties'][JSONSchema.BACKEND]['properties']['initial_layout'] = {
            'type': ['object', 'null'],
            'default': None,
        }

        # The same default and minimum as current RunConfig values
        self._schema['properties'][JSONSchema.BACKEND]['properties']['max_credits'] = {
            'type': 'integer',
            'default': 10,
            'minimum': 3,
            'maximum': 10,
        }

        # Timeout and wait are for remote backends where we have to connect over network
        if not is_local_backend(backend):
            self._schema['properties'][JSONSchema.BACKEND]['properties']['timeout'] = {
                "type": ["number", "null"],
                'default': None,
            }
            self._schema['properties'][JSONSchema.BACKEND]['properties']['wait'] = {
                'type': 'number',
                'default': 5.0,
                'minimum': 0.0,
            }

    def update_pluggable_schemas(self, input_parser):
        """
        Updates schemas of all pluggables

        Args:
            input_parser (obj): input parser
        """
        # find algorithm
        default_algo_name = self.get_property_default_value(PluggableType.ALGORITHM.value, JSONSchema.NAME)
        algo_name = input_parser.get_section_property(PluggableType.ALGORITHM.value, JSONSchema.NAME, default_algo_name)

        # update algorithm scheme
        if algo_name is not None:
            self._update_pluggable_schema(PluggableType.ALGORITHM.value, algo_name, default_algo_name)

        # update algorithm dependencies scheme
        config = {} if algo_name is None else get_pluggable_configuration(PluggableType.ALGORITHM, algo_name)
        classical = config.get('classical', False)
        # update algorithm backend from schema if it is classical or not
        if classical:
            if JSONSchema.BACKEND in self._schema['properties']:
                del self._schema['properties'][JSONSchema.BACKEND]
        else:
            if JSONSchema.BACKEND not in self._schema['properties']:
                self._schema['properties'][JSONSchema.BACKEND] = self._original_schema['properties'][JSONSchema.BACKEND]
                self.update_backend_schema(input_parser)

        pluggable_dependencies = config.get('depends', [])

        # remove pluggables from schema that are not in the dependencies of algorithm
        pluggable_dependency_names = [item['pluggable_type'] for item in pluggable_dependencies if 'pluggable_type' in item]
        for pluggable_type in local_pluggables_types():
            if pluggable_type not in [PluggableType.INPUT, PluggableType.ALGORITHM] and \
               pluggable_type.value not in pluggable_dependency_names and \
               pluggable_type.value in self._schema['properties']:
                del self._schema['properties'][pluggable_type.value]

        self._update_dependency_schemas(pluggable_dependencies, input_parser)

    def _update_dependency_schemas(self, pluggable_dependencies, input_parser):
        # update schema with dependencies recursevely
        for pluggable_type_dict in pluggable_dependencies:
            pluggable_type = pluggable_type_dict.get('pluggable_type')
            if pluggable_type is None:
                continue

            pluggable_name = None
            pluggable_defaults = pluggable_type_dict.get('default')
            default_properties = {}
            if pluggable_defaults is not None:
                for key, value in pluggable_defaults.items():
                    if key == JSONSchema.NAME:
                        pluggable_name = value
                    else:
                        default_properties[key] = value

            default_name = pluggable_name
            pluggable_name = input_parser.get_section_property(pluggable_type, JSONSchema.NAME, pluggable_name)
            if default_name is None:
                default_name = pluggable_name

            # update dependency schema
            self._update_pluggable_schema(pluggable_type, pluggable_name, default_name)
            for property_name in self._schema['properties'][pluggable_type]['properties'].keys():
                if property_name in default_properties:
                    self._schema['properties'][pluggable_type]['properties'][property_name]['default'] = default_properties[property_name]

            if pluggable_name is not None:
                config = get_pluggable_configuration(pluggable_type, pluggable_name)
                self._update_dependency_schemas(config.get('depends', []), input_parser)

    def _update_pluggable_schema(self, pluggable_type, pluggable_name, default_name):
        config = {}
        try:
            if pluggable_type is not None and pluggable_name is not None:
                config = get_pluggable_configuration(pluggable_type, pluggable_name)
        except:
            pass

        input_schema = config.get('input_schema', {})
        properties = input_schema.get('properties', {})
        properties[JSONSchema.NAME] = {'type': 'string'}
        required = input_schema.get('required', [])
        additionalProperties = input_schema.get('additionalProperties', True)
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
                raise AquaError("{}: Value '{}' is not of types: '{}'".format(section_name, value, types))

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
                raise AquaError("{}.{} Value '{}' is not of types: '{}'".format(
                    section_name, property_name, value, types))

        return value

    def validate(self, sections_json):
        try:
            logger.debug('Input: {}'.format(json.dumps(sections_json, sort_keys=True, indent=4)))
            logger.debug('Input Schema: {}'.format(json.dumps(self._schema, sort_keys=True, indent=4)))
            jsonschema.validate(sections_json, self._schema)
        except jsonschema.exceptions.ValidationError as ve:
            logger.info('JSON Validation error: {}'.format(str(ve)))
            raise AquaError(ve.message)

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
        config = get_pluggable_configuration(PluggableType.ALGORITHM, algo_name)
        if 'problems' in config:
            return config['problems']

        return []

    @staticmethod
    def get_value(value, types=None):
        """
        Returns a converted value based on schema types
        Args:
            value (obj): value
            type (array): schema types

        Returns:
            Returns converted value
        """
        types = types if types is not None else []
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
            raise AquaError("Empty section name.")

        return section_name

    @staticmethod
    def format_property_name(property_name):
        if property_name is None:
            property_name = ''
        property_name = property_name.strip()
        if len(property_name) == 0:
            raise AquaError("Empty property name.")

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

                resolved_ref = JSONSchema._resolve_schema_references(value, resolver)
                if resolved_ref:
                    schema[key] = resolved_ref

        elif isinstance(schema, list):
            for (idx, value) in enumerate(schema):
                resolved_ref = JSONSchema._resolve_schema_references(value, resolver)
                if resolved_ref:
                    schema[idx] = resolved_ref

        return schema
