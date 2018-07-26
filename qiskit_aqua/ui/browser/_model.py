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
from qiskit_aqua import (local_pluggables_types,
                         local_pluggables,
                         get_pluggable_configuration)
from qiskit_aqua.input import (local_inputs,
                               get_input_configuration)

class Model(object):
    
    _INPUT_NAME = 'Input'
    
    def __init__(self):
        """Create Model object."""
        
        self._property_titles = OrderedDict()
        self._sections = OrderedDict()
        self._sections[Model._INPUT_NAME] = OrderedDict()
        self._property_titles[Model._INPUT_NAME] = OrderedDict()
        for input_name in local_inputs():
            config = copy.deepcopy(get_input_configuration(input_name))
            self._populate_section(Model._INPUT_NAME,input_name,config)
            
        for pluggable_type in local_pluggables_types():
            self._sections[pluggable_type] = OrderedDict()
            self._property_titles[pluggable_type] = OrderedDict()
            for pluggable_name in local_pluggables(pluggable_type):
                config = copy.deepcopy(get_pluggable_configuration(pluggable_type,pluggable_name))
                self._populate_section(pluggable_type,pluggable_name,config)
            
    def _populate_section(self,section_type,section_name,configuration):
        self._sections[section_type][section_name] = OrderedDict()
        self._sections[section_type][section_name]['description'] = section_name
        self._sections[section_type][section_name]['properties'] = OrderedDict()
        self._sections[section_type][section_name]['other'] = OrderedDict()
        self._property_titles[section_type][section_name] = OrderedDict()
        for config_name,config_value in configuration.items():
            if config_name == 'description':
                self._sections[section_type][section_name]['description'] = str(config_value)
                continue;
                
            if config_name == 'input_schema':
                schema = configuration['input_schema']
                if 'properties' in schema:
                    for property,values in schema['properties'].items():  
                        if 'items' in values:
                            if 'type' in values['items']:
                                values['items'] = values['items']['type']
                        if 'oneOf' in values:
                            values['oneOf'] = values['oneOf'][0]
                            if 'enum' in values['oneOf']:
                                values['oneOf'] = values['oneOf']['enum']
                                
                            values['one of'] = values['oneOf']
                            del values['oneOf']
                                
                        self._sections[section_type][section_name]['properties'][property] = values
                        for k,v in values.items():
                            self._property_titles[section_type][section_name][k] = None
                continue
            
            self._sections[section_type][section_name]['other'][config_name] = config_value
           
        self._property_titles[section_type][section_name] = list(self._property_titles[section_type][section_name].keys())
        
        
    def top_names(self):
        return list(self._sections.keys())
    
    def get_section_description(self,top_name,section_name):
        return self._sections[top_name][section_name]['description']
    
    def get_property_titles(self,top_name,section_name):
        return self._property_titles[top_name][section_name]
            
    def get_sections(self):
        return self._sections
    
    def get_section_properties(self,top_name,section_name):
        return self._sections[top_name][section_name]['properties']
