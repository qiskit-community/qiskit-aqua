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

from abc import ABC, abstractmethod
import json
from collections import OrderedDict
import copy
import threading
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base GUI Model."""

    def __init__(self):
        """Create Model object."""
        self._parser = None
        self._custom_providers = {}
        self._available_providers = {}
        self._backendsthread = None
        self.get_available_providers()

    @property
    def providers(self):
        providers = copy.deepcopy(self._custom_providers)
        providers.update(self._available_providers)
        return providers

    def get_available_providers(self):
        from qiskit.aqua import register_ibmq_and_get_known_providers
        if self._backendsthread is not None:
            return

        self._register_ibmq_and_get_known_providers = register_ibmq_and_get_known_providers
        self._backendsthread = threading.Thread(target=self._get_available_providers,
                                                name='Available providers')
        self._backendsthread.daemon = True
        self._backendsthread.start()

    def _get_available_providers(self):
        try:
            self._available_providers = OrderedDict([x for x in
                                                     self._register_ibmq_and_get_known_providers().items() if len(x[1]) > 0])
        except Exception as e:
            logger.debug(str(e))
        finally:
            self._backendsthread = None

    def is_empty(self):
        return self._parser is None or len(self._parser.get_section_names()) == 0

    @abstractmethod
    def new(self, parser_class, template_file, populate_defaults):
        try:
            dict = {}
            jsonfile = template_file
            with open(jsonfile) as json_file:
                dict = json.load(json_file)

            self._parser = parser_class(dict)
            self._parser.parse()
            if populate_defaults:
                self._parser.validate_merge_defaults()
                self._parser.commit_changes()

            return self._parser.get_section_names()
        except:
            self._parser = None
            raise

    @abstractmethod
    def load_file(self, filename, parser_class, populate_defaults):
        from qiskit.aqua.parser import JSONSchema
        from qiskit.aqua import get_provider_from_backend, get_backends_from_provider
        if filename is None:
            return []
        try:
            self._parser = parser_class(filename)
            self._parser.parse()
            # before merging defaults attempts to find a provider for the backend
            provider = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
            if provider is None:
                backend_name = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
                if backend_name is not None:
                    self._parser.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER, get_provider_from_backend(backend_name))
            else:
                try:
                    if provider not in self.providers:
                        self._custom_providers[provider] = get_backends_from_provider(provider)
                except Exception as e:
                    logger.debug(str(e))
        except:
            self._parser = None
            raise

        try:
            if populate_defaults:
                self._parser.validate_merge_defaults()

            return self.get_section_names()
        except:
            raise
        finally:
            self._parser.commit_changes()

    def get_filename(self):
        if self._parser is None:
            return None

        return self._parser.get_filename()

    def is_modified(self):
        if self._parser is None:
            return False

        return self._parser.is_modified()

    def save_to_file(self, filename):
        if self.is_empty():
            raise Exception("Empty input data.")

        self._parser.save_to_file(filename)

    def get_section_names(self):
        if self._parser is None:
            return []

        return self._parser.get_section_names()

    def get_property_default_values(self, section_name, property_name):
        if self._parser is None:
            return None

        return self._parser.get_property_default_values(section_name, property_name)

    def section_is_text(self, section_name):
        if self._parser is None:
            return False

        return self._parser.section_is_text(section_name)

    def get_section(self, section_name):
        return self._parser.get_section(section_name) if self._parser is not None else None

    def get_section_text(self, section_name):
        if self._parser is None:
            return ''

        return self._parser.get_section_text(section_name)

    def get_section_properties(self, section_name):
        if self._parser is None:
            return {}

        return self._parser.get_section_properties(section_name)

    @abstractmethod
    def default_properties_equals_properties(self, section_name):
        pass

    def get_section_property(self, section_name, property_name):
        if self._parser is None:
            return None

        return self._parser.get_section_property(section_name, property_name)

    def set_section(self, section_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.set_section(section_name)
        value = self._parser.get_section_default_properties(section_name)
        if isinstance(value, dict):
            for property_name, property_value in value.items():
                self._parser.set_section_property(section_name, property_name, property_value)

            # do one more time in case schema was updated
            value = self._parser.get_section_default_properties(section_name)
            for property_name, property_value in value.items():
                self._parser.set_section_property(section_name, property_name, property_value)
        else:
            if value is None:
                types = self._parser.get_section_types(section_name)
                if 'null' not in types:
                    if 'string' in types:
                        value = ''
                    elif 'object' in types:
                        value = {}
                    elif 'array' in types:
                        value = []

            self._parser.set_section_data(section_name, value)

    def set_default_properties_for_name(self, section_name):
        from qiskit.aqua.parser import JSONSchema
        if self._parser is None:
            raise Exception('Input not initialized.')

        name = self._parser.get_section_property(section_name, JSONSchema.NAME)
        self._parser.delete_section_properties(section_name)
        value = self._parser.get_section_default_properties(section_name)
        if JSONSchema.BACKEND != section_name and name is not None:
            self._parser.set_section_property(section_name, JSONSchema.NAME, name)
        if isinstance(value, dict):
            for property_name, property_value in value.items():
                if JSONSchema.BACKEND == section_name or property_name != JSONSchema.NAME:
                    self._parser.set_section_property(section_name, property_name, property_value)
        else:
            if value is None:
                types = self._parser.get_section_types(section_name)
                if 'null' not in types:
                    if 'string' in types:
                        value = ''
                    elif 'object' in types:
                        value = {}
                    elif 'array' in types:
                        value = []

            self._parser.set_section_data(section_name, value)

    @staticmethod
    def is_pluggable_section(section_name):
        from qiskit.aqua.parser import BaseParser
        return BaseParser.is_pluggable_section(section_name)

    def get_pluggable_section_names(self, section_name):
        from qiskit.aqua.parser import BaseParser
        from qiskit.aqua import PluggableType, local_pluggables
        from qiskit.aqua.parser import JSONSchema
        if not BaseModel.is_pluggable_section(section_name):
            return []

        if PluggableType.ALGORITHM.value == section_name:
            problem_name = None
            if self._parser is not None:
                problem_name = self.get_section_property(JSONSchema.PROBLEM, JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(JSONSchema.PROBLEM, JSONSchema.NAME)

            if problem_name is None:
                return local_pluggables(PluggableType.ALGORITHM)

            algo_names = []
            for algo_name in local_pluggables(PluggableType.ALGORITHM):
                problems = BaseParser.get_algorithm_problems(algo_name)
                if problem_name in problems:
                    algo_names.append(algo_name)

            return algo_names

        return local_pluggables(section_name)

    def delete_section(self, section_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.delete_section(section_name)

    def get_default_sections(self):
        if self._parser is None:
            raise Exception('Input not initialized.')

        return self._parser.get_default_sections()

    def get_section_default_properties(self, section_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        return self._parser.get_section_default_properties(section_name)

    def allows_additional_properties(self, section_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        return self._parser.allows_additional_properties(section_name)

    def get_property_default_value(self, section_name, property_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        return self._parser.get_property_default_value(section_name, property_name)

    def get_property_types(self, section_name, property_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        return self._parser.get_property_types(section_name, property_name)

    def set_section_property(self, section_name, property_name, value):
        from qiskit.aqua.parser import BaseParser
        from qiskit.aqua.parser import JSONSchema
        from qiskit.aqua import get_backends_from_provider
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.set_section_property(section_name, property_name, value)
        if property_name == JSONSchema.NAME and BaseParser.is_pluggable_section(section_name):
            properties = self._parser.get_section_default_properties(section_name)
            if isinstance(properties, dict):
                properties[JSONSchema.NAME] = value
                self._parser.delete_section_properties(section_name)
                for property_name, property_value in properties.items():
                    self._parser.set_section_property(section_name, property_name, property_value)
        elif section_name == JSONSchema.BACKEND and property_name == JSONSchema.PROVIDER:
            backends = get_backends_from_provider(value)
            if value not in self.providers:
                self._custom_providers[value] = backends

            backend = backends[0] if len(backends) > 0 else ''
            self._parser.set_section_property(section_name, JSONSchema.NAME, backend)

    def delete_section_property(self, section_name, property_name):
        from qiskit.aqua.parser import BaseParser
        from qiskit.aqua.parser import JSONSchema
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.delete_section_property(section_name, property_name)
        if property_name == JSONSchema.NAME and BaseParser.is_pluggable_section(section_name):
            self._parser.delete_section_properties(section_name)
        elif section_name == JSONSchema.BACKEND and (property_name == JSONSchema.PROVIDER or property_name == JSONSchema.NAME):
            self._parser.delete_section_properties(section_name)

    def set_section_text(self, section_name, value):
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.set_section_data(section_name, value)

    def delete_section_text(self, section_name):
        if self._parser is None:
            raise Exception('Input not initialized.')

        self._parser.delete_section_text(section_name)
