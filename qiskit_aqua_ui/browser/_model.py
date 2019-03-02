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

from collections import OrderedDict
import copy


class Model(object):

    def __init__(self):
        """Create Model object."""
        self._data_loaded = False

    def _load_data(self):
        if self._data_loaded:
            return

        from qiskit.aqua import (local_pluggables_types,
                                 local_pluggables,
                                 get_pluggable_configuration)

        self._schema_property_titles = OrderedDict()
        self._sections = OrderedDict()
        for pluggable_type in local_pluggables_types():
            self._sections[pluggable_type.value] = OrderedDict()
            self._schema_property_titles[pluggable_type.value] = OrderedDict()
            for pluggable_name in local_pluggables(pluggable_type):
                config = copy.deepcopy(get_pluggable_configuration(pluggable_type, pluggable_name))
                self._populate_section(pluggable_type.value, pluggable_name, config)

        self._data_loaded = True

    def _populate_section(self, pluggable_type, pluggable_name, configuration):
        self._sections[pluggable_type][pluggable_name] = OrderedDict()
        self._sections[pluggable_type][pluggable_name]['description'] = pluggable_name
        self._sections[pluggable_type][pluggable_name]['properties'] = OrderedDict()
        self._sections[pluggable_type][pluggable_name]['problems'] = []
        self._sections[pluggable_type][pluggable_name]['depends'] = OrderedDict()
        self._schema_property_titles[pluggable_type][pluggable_name] = OrderedDict()
        for config_name, config_value in configuration.items():
            if config_name == 'description':
                self._sections[pluggable_type][pluggable_name]['description'] = str(config_value)
                continue

            if config_name == 'problems' and isinstance(config_value, list):
                self._sections[pluggable_type][pluggable_name]['problems'] = config_value
                continue

            if config_name == 'depends' and isinstance(config_value, list):
                self._sections[pluggable_type][pluggable_name]['depends'] = config_value
                continue

            if config_name == 'input_schema' and isinstance(config_value, dict):
                schema = config_value
                if 'properties' in schema:
                    for property, values in schema['properties'].items():
                        if 'items' in values:
                            if 'type' in values['items']:
                                values['items'] = values['items']['type']
                        if 'oneOf' in values:
                            values['oneOf'] = values['oneOf'][0]
                            if 'enum' in values['oneOf']:
                                values['oneOf'] = values['oneOf']['enum']

                            values['one of'] = values['oneOf']
                            del values['oneOf']

                        self._sections[pluggable_type][pluggable_name]['properties'][property] = values
                        for k, v in values.items():
                            self._schema_property_titles[pluggable_type][pluggable_name][k] = None
                continue

        self._schema_property_titles[pluggable_type][pluggable_name] = list(self._schema_property_titles[pluggable_type][pluggable_name].keys())

    def pluggable_names(self):
        self._load_data()
        return list(self._sections.keys())

    def get_pluggable_description(self, pluggable_type, pluggable_name):
        self._load_data()
        return self._sections[pluggable_type][pluggable_name]['description']

    def get_pluggable_problems(self, pluggable_type, pluggable_name):
        self._load_data()
        return self._sections[pluggable_type][pluggable_name]['problems']

    def get_pluggable_dependency(self, pluggable_type, pluggable_name, dependency_type):
        self._load_data()
        depends = self._sections[pluggable_type][pluggable_name]['depends']
        for dependency in depends:
            if dependency.get('pluggable_type') == dependency_type:
                return dependency

        return {}

    def get_pluggable_schema_property_titles(self, pluggable_type, pluggable_name):
        self._load_data()
        return self._schema_property_titles[pluggable_type][pluggable_name]

    def get_sections(self):
        self._load_data()
        return self._sections

    def get_pluggable_schema_properties(self, pluggable_type, pluggable_name):
        self._load_data()
        return self._sections[pluggable_type][pluggable_name]['properties']
