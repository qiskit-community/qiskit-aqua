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

from qiskit_aqua_chemistry import AquaChemistryError
from qiskit_aqua_chemistry.drivers import ConfigurationManager
import json
import os
from collections import OrderedDict
import logging
import copy
import pprint
from qiskit_aqua import (local_pluggables_types,
                         get_algorithm_configuration,
                         local_algorithms)
from qiskit_aqua.parser import JSONSchema
from qiskit_aqua_chemistry.core import (local_chemistry_operators, get_chemistry_operator_configuration)

logger = logging.getLogger(__name__)

class InputParser(object):
    """Common input file parser."""

    OPERATOR = 'operator'
    DRIVER = 'driver'
    AUTO_SUBSTITUTIONS = 'auto_substitutions'
    _OLD_ENABLE_SUBSTITUTIONS = 'enable_substitutions'
 
    _START_COMMENTS = ['#','%']
    _START_SECTION = '&'
    _END_SECTION = '&end'
    _PROPVALUE_SEPARATOR = '='
    
    _OPTIMIZER = 'optimizer'
    _VARIATIONAL_FORM = 'variational_form'
    _UNKNOWN = 'unknown'
    _HDF5_INPUT = 'hdf5_input'
    _DRIVER_NAMES = None
    _PROPERTY_ORDER = [JSONSchema.NAME,_UNKNOWN]
    
    def __init__(self, input=None):
        """Create InputParser object."""
        self._sections = OrderedDict()
        self._original_sections = OrderedDict()
        self._filename = None
        self._inputdict = None
        if input is not None:
            if isinstance(input, dict):
                self._inputdict = input
            elif isinstance(input, str):
                self._filename = input
            else:
                raise AquaChemistryError("Invalid parser input type.")
             
        self._section_order = [JSONSchema.NAME,JSONSchema.PROBLEM,
                               InputParser.DRIVER,InputParser._UNKNOWN,
                               InputParser.OPERATOR,JSONSchema.ALGORITHM]
        for pluggable_type in local_pluggables_types():
            if pluggable_type != JSONSchema.ALGORITHM:
                self._section_order.append(pluggable_type)
                
        self._section_order.append(JSONSchema.BACKEND)
        
        jsonfile = os.path.join(os.path.dirname(__file__), 'substitutions.json')
        with open(jsonfile) as json_file:
            self._substitutions = json.load(json_file)
            
        self._json_schema = JSONSchema(os.path.join(os.path.dirname(__file__), 'input_schema.json'))
        
        # get some properties from algorithms schema
        self._json_schema.copy_section_from_aqua_schema(JSONSchema.ALGORITHM)
        self._json_schema.copy_section_from_aqua_schema(JSONSchema.BACKEND)
        self._json_schema.copy_section_from_aqua_schema(JSONSchema.PROBLEM)
        self._json_schema.schema['properties'][JSONSchema.PROBLEM]['properties'][InputParser.AUTO_SUBSTITUTIONS] = {
                   "type": "boolean",
                   "default": "true"
        }
        self._json_schema.populate_problem_names()
            
        self._json_schema.commit_changes()
        #logger.debug('Resolved Schema Input: {}'.format(json.dumps(self._json_schema.schema, sort_keys=True, indent=4)))
                    
    def _order_sections(self,sections):
        sections_sorted = OrderedDict(sorted(list(sections.items()),
             key=lambda x: self._section_order.index(x[0]) 
             if x[0] in self._section_order else self._section_order.index(InputParser._UNKNOWN)))
        
        for section,values in sections_sorted.items():
            if not self.section_is_driver(section) and 'properties' in values and isinstance(values['properties'],dict):
                sections_sorted[section]['properties'] = OrderedDict(sorted(list(values['properties'].items()),
                             key=lambda x: InputParser._PROPERTY_ORDER.index(x[0]) 
                             if x[0] in InputParser._PROPERTY_ORDER else InputParser._PROPERTY_ORDER.index(InputParser._UNKNOWN)))
           
        return sections_sorted
        
    def parse(self):
        """Parse the data."""
        if self._inputdict is None:
            if self._filename is None:
                raise AquaChemistryError("Missing input file")
                
            section = None
            self._sections = OrderedDict()
            with open(self._filename, 'rt', encoding="utf8", errors='ignore') as f:
                for line in f:
                    section = self._process_line(section,line)
        else:
            self._load_parser_from_dict()
            
        # check for old enable_substitutions name
        old_enable_substitutions = self.get_section_property(JSONSchema.PROBLEM, InputParser._OLD_ENABLE_SUBSTITUTIONS)
        if old_enable_substitutions is not None:
            self.delete_section_property(JSONSchema.PROBLEM, InputParser._OLD_ENABLE_SUBSTITUTIONS)
            self.set_section_property(JSONSchema.PROBLEM, InputParser.AUTO_SUBSTITUTIONS,old_enable_substitutions)
                
        self._json_schema.update_pluggable_input_schemas(self)
        self._update_driver_input_schemas()
        self._update_operator_input_schema()
        self._sections = self._order_sections(self._sections)  
        self._original_sections = copy.deepcopy(self._sections)
        
    def _load_parser_from_dict(self):
        self._sections = OrderedDict()
        for section_name,value in self._inputdict.items():
            section_name = JSONSchema.format_section_name(section_name).lower()
            self._sections[section_name] = OrderedDict()
            self._sections[section_name]['properties'] = OrderedDict()
            self._sections[section_name]['data'] = ''
            if isinstance(value, dict):
                for k,v in value.items():
                    self._sections[section_name]['properties'][k] = v
                contents = ''
                properties = self._sections[section_name]['properties']
                lastIndex = len(properties) - 1
                for i,(k,v) in enumerate(properties.items()):
                    contents += '{}{}{}'.format(k,InputParser._PROPVALUE_SEPARATOR,v)
                    if i < lastIndex:
                        contents += '\n'
                self._sections[section_name]['data'] = contents
            elif isinstance(value, list) or isinstance(value, str):
                lines = []
                if isinstance(value, list):
                    lines = value
                    self._sections[section_name]['data'] = '\n'.join(str(e) for e in value)
                else:
                    lines  = value.splitlines()
                    self._sections[section_name]['data'] = value
            
                for line in lines:
                    k,v = self._get_key_value(line)
                    if k is not None and v is not None:   
                        self._sections[section_name]['properties'][k] = v
            else:
                raise AquaChemistryError("Invalid parser input type for section {}".format(section_name))
       
    def is_modified(self):
        """
        Returns true if data has been changed
        """
        original_section_names = set(self._original_sections.keys())
        section_names = set(self._sections.keys())
        if original_section_names != section_names:
            return True
        
        for section_name in section_names:
            original_section = self._original_sections[section_name]
            section = self._sections[section_name]
            if self.section_is_text(section_name):
                original_data = original_section['data'] if 'data' in original_section else None
                data = section['data'] if 'data' in section else None
                if original_data != data:
                    return True
            else:
                original_properties = original_section['properties'] if 'properties' in original_section else None
                properties = section['properties'] if 'properties' in section else None
                if original_properties != properties:
                    return True
        
        return False
            
    @staticmethod
    def is_pluggable_section(section_name):
        return JSONSchema.format_section_name(section_name).lower() in local_pluggables_types()
    
    def get_section_types(self,section_name):
       return self._json_schema.get_section_types(section_name)
    
    def get_property_types(self,section_name,property_name):
        return self._json_schema.get_property_types(section_name,property_name)
    
    def get_default_sections(self):
        properties = self._json_schema.get_default_sections()
        driver_name = self.get_section_property(InputParser.DRIVER,JSONSchema.NAME)
        if driver_name is not None:
             properties[driver_name.lower()] = { 
                     "type": "object"
             }
        return properties
    
    def get_default_section_names(self):
        sections = self.get_default_sections()
        return list(sections.keys()) if sections is not None else []
    
    def get_section_default_properties(self,section_name):
        return self._json_schema.get_section_default_properties(section_name)
     
    def allows_additional_properties(self,section_name):
        return self._json_schema.allows_additional_properties(section_name)
        
    def get_property_default_values(self,section_name,property_name):
        return self._json_schema.get_property_default_values(section_name,property_name)
    
    def get_property_default_value(self,section_name,property_name):
        return self._json_schema.get_property_default_value(section_name,property_name)
    
    def get_filename(self):
        """Return the filename."""
        return self._filename
    
    @staticmethod
    def get_operator_problems(input_name):
        config = get_chemistry_operator_configuration(input_name)
        if 'problems' in config:
            return config['problems']
        
        return []
    
    @staticmethod
    def get_algorithm_problems(algo_name):
        return JSONSchema.get_algorithm_problems(algo_name)  
        
    def _update_operator_input_schema(self):
        # find operator
        default_name = self.get_property_default_value(InputParser.OPERATOR,JSONSchema.NAME)
        operator_name = self.get_section_property(InputParser.OPERATOR,JSONSchema.NAME,default_name)
        if operator_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(JSONSchema.PROBLEM,JSONSchema.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(JSONSchema.PROBLEM,JSONSchema.NAME)
                
            if problem_name is None:
                raise AquaChemistryError("No algorithm 'problem' section found on input.")
        
            for name in local_chemistry_operators():
                if problem_name in self.get_operator_problems(name):
                    # set to the first input to solve the problem
                    operator_name = name
                    break
                
        if operator_name is None:
            # just remove fromm schema if none solves the problem
            if InputParser.OPERATOR in self._json_schema.schema['properties']:
                del self._json_schema.schema['properties'][InputParser.OPERATOR]
            
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
        properties[JSONSchema.NAME] = { 'type': 'string' }
        required = input_schema['required'] if 'required' in input_schema else [] 
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True 
        if default_name is not None:
            properties[JSONSchema.NAME]['default'] = default_name
            required.append(JSONSchema.NAME) 
        
        if InputParser.OPERATOR not in self._json_schema.schema['properties']:
            self._json_schema.schema['properties'][InputParser.OPERATOR] = { 'type': 'object' }
        
        self._json_schema.schema['properties'][InputParser.OPERATOR]['properties'] = properties
        self._json_schema.schema['properties'][InputParser.OPERATOR]['required'] = required
        self._json_schema.schema['properties'][InputParser.OPERATOR]['additionalProperties'] = additionalProperties
        
    def _merge_dependencies(self):
        algo_name = self.get_section_property(JSONSchema.ALGORITHM,JSONSchema.NAME)
        if algo_name is None:
            return
        
        config = get_algorithm_configuration(algo_name)
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {} if 'defaults' not in config else config['defaults']
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
                for key,value in pluggable_defaults[pluggable_type].items():
                    if key == JSONSchema.NAME:
                        pluggable_name = pluggable_defaults[pluggable_type][key]
                    else:
                        new_properties[key] = value
                        
            if pluggable_name is None:
                continue
            
            if pluggable_type not in section_names:
                self.set_section(pluggable_type)
                
            if self.get_section_property(pluggable_type,JSONSchema.NAME) is None:
                self.set_section_property(pluggable_type,JSONSchema.NAME,pluggable_name)
               
            if pluggable_name == self.get_section_property(pluggable_type,JSONSchema.NAME):
                properties = self.get_section_properties(pluggable_type)
                if new_properties:
                    new_properties.update(properties)
                else:
                    new_properties = properties
                    
                self.set_section_properties(pluggable_type,new_properties)
       
    def _update_driver_input_schemas(self):
        driver_name = self.get_section_property(InputParser.DRIVER,JSONSchema.NAME)
        if driver_name is not None:
             driver_name = driver_name.strip().lower()
             
        mgr =  ConfigurationManager()
        configs = mgr.configurations
        for (name,config) in configs.items():
            name = name.lower()
            if driver_name is not None and driver_name == name:
                input_schema = copy.deepcopy(config['input_schema']) if 'input_schema' in config else { 'type': 'object'}
                if '$schema' in input_schema:
                    del input_schema['$schema']
                if 'id' in input_schema:
                    del input_schema['id']
                    
                self._json_schema.schema['properties'][driver_name] = input_schema
            else:
                if name in self._json_schema.schema['properties']:
                    del self._json_schema.schema['properties'][name]
                    
    @staticmethod
    def _load_driver_names():
        if InputParser._DRIVER_NAMES is None:
            mgr =  ConfigurationManager()
            InputParser._DRIVER_NAMES = [name.lower() for name in mgr.module_names]
            
    def _merge_default_values(self):
        section_names = self.get_section_names()
        if JSONSchema.NAME not in section_names:
            self.set_section(JSONSchema.NAME)
            
        if JSONSchema.ALGORITHM in section_names:
            if JSONSchema.PROBLEM not in section_names:
                self.set_section(JSONSchema.PROBLEM)
                
        self._json_schema.update_pluggable_input_schemas(self)
        self._merge_dependencies()
        self._update_driver_sections()
        self._update_driver_input_schemas()
        self._update_operator_input_schema()
      
        section_names = set(self.get_section_names()) | set(self.get_default_section_names())
        for section_name in section_names:
            if section_name not in self._sections:
                self.set_section(section_name)
           
            new_properties = self.get_section_default_properties(section_name)
            if new_properties is not None:
                if self.section_is_text(section_name):
                    text = self.get_section_text(section_name)
                    if (text is None or len(text) == 0) and \
                        isinstance(new_properties,str) and \
                        len(new_properties) > 0 and \
                        text != new_properties:
                        self.set_section_data(section_name,new_properties)
                else:
                    properties = self.get_section_properties(section_name)
                    new_properties.update(properties)
                    self.set_section_properties(section_name,new_properties)
                    
        self._sections = self._order_sections(self._sections)
          
    def validate_merge_defaults(self):
        self._merge_default_values()
        self._json_schema.validate(self.to_JSON())
        self._validate_algorithm_problem()
        self._validate_operator_problem()
            
    def _validate_algorithm_problem(self):
        algo_name = self.get_section_property(JSONSchema.ALGORITHM,JSONSchema.NAME)
        if algo_name is None:
            return
        
        problem_name = self.get_section_property(JSONSchema.PROBLEM,JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM,JSONSchema.NAME)
                
        if problem_name is None:
            raise AquaChemistryError("No algorithm 'problem' section found on input.")
       
        problems = InputParser.get_algorithm_problems(algo_name)
        if problem_name not in problems:
            raise AquaChemistryError(
            "Problem: {} not in the list of problems: {} for algorithm: {}.".format(problem_name,problems,algo_name))
        
    def _validate_operator_problem(self):
        operator_name = self.get_section_property(InputParser.OPERATOR,JSONSchema.NAME)
        if operator_name is None:
            return
        
        problem_name = self.get_section_property(JSONSchema.PROBLEM,JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM,JSONSchema.NAME)
                
        if problem_name is None:
            raise AquaChemistryError("No algorithm 'problem' section found on input.")
       
        problems = InputParser.get_operator_problems(operator_name)
        if problem_name not in problems:
            raise AquaChemistryError(
            "Problem: {} not in the list of problems: {} for operator: {}.".format(problem_name,problems,operator_name))
           
    def to_JSON(self):
        json_dict = OrderedDict()
        for section_name in self.get_section_names():
            if self.section_is_text(section_name):
                json_dict[section_name] = self.get_section_text(section_name)
            else:
                json_dict[section_name] = self.get_section_properties(section_name) 
                
        return json_dict
    
    def to_dictionary(self):
        dict = OrderedDict()
        for section_name in self.get_section_names():
            if self.section_is_text(section_name):
                dict[section_name] = self.get_section_text(section_name).splitlines()
            else:
                dict[section_name] = self.get_section_properties(section_name) 
     
        return dict
    
    def commit_changes(self):
        self._original_sections = copy.deepcopy(self._sections)
                    
    def save_to_file(self,file_name):
        if file_name is None:
            raise AquaChemistryError('Missing file path')
            
        file_name = file_name.strip()
        if len(file_name) == 0:
            raise AquaChemistryError('Missing file path')
            
        prev_filename = self.get_filename()
        sections = copy.deepcopy(self.get_sections())
        if prev_filename is not None:
            prev_dirname = os.path.dirname(os.path.realpath(prev_filename))
            dirname = os.path.dirname(os.path.realpath(file_name))
            if prev_dirname != dirname:
                InputParser._from_relative_to_abs_paths(sections,prev_filename)
                
        contents = ''
        lastIndex = len(sections) - 1
        for i,(section_name,section) in enumerate(sections.items()):
            contents += '{}{}'.format(InputParser._START_SECTION,section_name)
            if self.section_is_text(section_name):
                value = section['data']
                if value is not None:
                    contents += '\n{}'.format(str(value))
            else:
                if 'properties' in section:
                    for k,v in section['properties'].items():
                        contents += '\n   {}{}{}'.format(k,InputParser._PROPVALUE_SEPARATOR,str(v))
                    
            contents += '\n{}'.format(InputParser._END_SECTION)
            if i < lastIndex:
                contents += '\n\n'
      
        with open(file_name, 'w') as f:
            print(contents, file=f)
            
    def export_dictionary(self,file_name):
        if file_name is None:
            raise AquaChemistryError('Missing file path')
            
        file_name = file_name.strip()
        if len(file_name) == 0:
            raise AquaChemistryError('Missing file path')
            
        value = json.loads(json.dumps(self.to_dictionary()))
        value = pprint.pformat(value, indent=4)
        with open(file_name, 'w') as f:
            print(value, file=f)
            
    @staticmethod
    def _from_relative_to_abs_paths(sections,filename):
        directory = os.path.dirname(filename)
        for _,section in sections.items():
            if 'properties' in section:
                for key,value in section['properties'].items():
                    if key == InputParser._HDF5_INPUT:
                        if value is not None and not os.path.isabs(value):
                            value = os.path.abspath(os.path.join(directory,value))
                            InputParser._set_section_property(sections,section[JSONSchema.NAME],key,value,['string'])
            
    def section_is_driver(self,section_name):
        section_name = JSONSchema.format_section_name(section_name).lower()
        InputParser._load_driver_names()
        return section_name in InputParser._DRIVER_NAMES
    
    def section_is_text(self,section_name):
        section_name = JSONSchema.format_section_name(section_name).lower()
        types = self.get_section_types(section_name)
        if len(types) > 0: 
            return 'string' in types
        
        return False
               
    def get_sections(self):
        return self._sections
                
    def get_section(self, section_name):
        """Return a Section by name.
        Args:
            section_name (str): the name of the section, case insensitive
        Returns:
            Section: The section with this name
        Raises:
            AquaChemistryError: if the section does not exist.
        """
        section_name = JSONSchema.format_section_name(section_name).lower()     
        try:
            return self._sections[section_name]
        except KeyError:
            raise AquaChemistryError('No section "{0}"'.format(section_name))
            
    def get_section_text(self,section_name):
        section = self.get_section(section_name)
        if section is None:
            return ''
        
        if 'data' in section:
            return section['data']
        
        return ''
     
    def get_section_properties(self,section_name):
        section = self.get_section(section_name)
        if section is None:
            return {}
        
        if 'properties' in section:
            return section['properties']
        
        return {}
           
    def get_section_property(self, section_name, property_name, default_value = None):
        """Return a property by name.
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            default_value : default value in case it is not found
        Returns:
            Value: The property value
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_name = JSONSchema.format_property_name(property_name)
        if section_name in self._sections:
            section = self._sections[section_name]
            if 'properties' in section and property_name in section['properties']:
                return section['properties'][property_name]
            
        return default_value
    
    def get_section_data(self, section_name, default_value = None):
        """
        Return a section data.
        Args:
            section_name (str): the name of the section, case insensitive
            default_value : default value in case it is not found
        Returns:
            Value: data value
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        if section_name in self._sections:
            section = self._sections[section_name]
            if 'data' in section:
                return section['data']
            
        return default_value
    
    def set_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        if section_name not in self._sections:
            self._sections[section_name] = OrderedDict([(JSONSchema.NAME,section_name)])
            self._sections[section_name]['properties'] = OrderedDict()
            self._sections[section_name]['data'] = ''
            self._sections = self._order_sections(self._sections)
            
    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        if section_name not in self._sections:
            return
            
        del self._sections[section_name]
            
        # update schema
        self._json_schema.rollback_changes()
        self._json_schema.update_pluggable_input_schemas(self)
        self._update_driver_input_schemas()
        self._update_operator_input_schema()
            
    def set_section_properties(self, section_name, properties):
        self.delete_section_properties(section_name)
        for property_name,value in properties.items():
            self.set_section_property(section_name,property_name,value)
    
    def set_section_property(self, section_name, property_name, value):
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_name = JSONSchema.format_property_name(property_name)
        value = self._json_schema.check_property_value(section_name, property_name, value)
        types = self.get_property_types(section_name,property_name) 
       
        parser_temp = copy.deepcopy(self)
        InputParser._set_section_property(parser_temp._sections,section_name,property_name,value, types)
        msg = self._json_schema.validate_property(parser_temp.to_JSON(),section_name, property_name)
        if msg is not None:
            raise AquaChemistryError("{}.{}: Value '{}': '{}'".format(section_name, property_name, value, msg))
 
        InputParser._set_section_property(self._sections,section_name,property_name,value, types)
        if property_name == JSONSchema.NAME:
            if InputParser.OPERATOR == section_name:
                self._update_operator_input_schema()
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties,dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name,property_name)
            elif JSONSchema.PROBLEM == section_name:
                self._update_algorithm_problem()
                self._update_operator_problem()
            elif InputParser.is_pluggable_section(section_name):
                self._json_schema.update_pluggable_input_schemas(self)
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties,dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != JSONSchema.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name,property_name)
                
                if section_name == JSONSchema.ALGORITHM:
                    self._update_dependency_sections()
            elif value is not None:
                value = str(value).lower().strip()
                if len(value) > 0 and self.section_is_driver(value):
                    self._update_driver_input_schemas()
                    self._update_driver_sections()
                    
        self._sections = self._order_sections(self._sections)
        
    def _update_algorithm_problem(self):
        problem_name = self.get_section_property(JSONSchema.PROBLEM,JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM,JSONSchema.NAME)
                
        if problem_name is None:
            raise AquaChemistryError("No algorithm 'problem' section found on input.")
            
        algo_name = self.get_section_property(JSONSchema.ALGORITHM,JSONSchema.NAME)
        if algo_name is not None and problem_name in InputParser.get_algorithm_problems(algo_name):
            return
        
        for algo_name in local_algorithms():
            if problem_name in self.get_algorithm_problems(algo_name):
                # set to the first algorithm to solve the problem
                self.set_section_property(JSONSchema.ALGORITHM,JSONSchema.NAME,algo_name)
                return
            
        # no algorithm solve this problem, remove section
        self.delete_section(JSONSchema.ALGORITHM)
        
    def _update_operator_problem(self):
        problem_name = self.get_section_property(JSONSchema.PROBLEM,JSONSchema.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(JSONSchema.PROBLEM,JSONSchema.NAME)
                
        if problem_name is None:
            raise AquaChemistryError("No algorithm 'problem' section found on input.")
            
        operator_name = self.get_section_property(InputParser.OPERATOR,JSONSchema.NAME)
        if operator_name is not None and problem_name in InputParser.get_operator_problems(operator_name):
            return
        
        for operator_name in local_chemistry_operators():
            if problem_name in self.get_operator_problems(operator_name):
                # set to the first input to solve the problem
                self.set_section_property(InputParser.OPERATOR,JSONSchema.NAME,operator_name)
                return
            
        # no input solve this problem, remove section
        self.delete_section(InputParser.OPERATOR)
                    
    def _update_dependency_sections(self):
        algo_name = self.get_section_property(JSONSchema.ALGORITHM,JSONSchema.NAME)
        config = {} if algo_name is None else get_algorithm_configuration(algo_name) 
        classical = config['classical'] if 'classical' in config else False 
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {} if 'defaults' not in config else config['defaults']
        pluggable_types = local_pluggables_types()
        for pluggable_type in pluggable_types:
            if pluggable_type != JSONSchema.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from input that are not in the dependencies
                if pluggable_type in self._sections:
                   del self._sections[pluggable_type] 
       
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            if pluggable_type in pluggable_defaults:
                if JSONSchema.NAME in pluggable_defaults[pluggable_type]:
                    pluggable_name = pluggable_defaults[pluggable_type][JSONSchema.NAME]
           
            if pluggable_name is not None and pluggable_type not in self._sections:
                self.set_section_property(pluggable_type,JSONSchema.NAME,pluggable_name)
                
        # update backend based on classical
        if classical:
            if JSONSchema.BACKEND in self._sections:
                del self._sections[JSONSchema.BACKEND]
        else:
            if JSONSchema.BACKEND not in self._sections:
                self.set_section_properties(JSONSchema.BACKEND,self.get_section_default_properties(JSONSchema.BACKEND))
                
    def _update_driver_sections(self):
        driver_name = self.get_section_property(InputParser.DRIVER,JSONSchema.NAME)
        if driver_name is not None:
             driver_name = driver_name.strip().lower()
             
        mgr =  ConfigurationManager()
        configs = mgr.configurations
        for (name,config) in configs.items():
            name = name.lower()
            if driver_name is not None and driver_name == name:
                continue
            
            if name in self._sections:
                del self._sections[name]
            
        if driver_name is not None and driver_name not in self._sections:
            self.set_section(driver_name)
            value = self.get_section_default_properties(driver_name)
            if isinstance(value,dict):
                for property_name,property_value in value.items():
                    self.set_section_property(driver_name,property_name,property_value)
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
                    
                self.set_section_data(driver_name,value)
                    
    @staticmethod
    def _set_section_property(sections, section_name, property_name, value, types):
        """
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            value : property value
            types : schema valid types
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_name = JSONSchema.format_property_name(property_name)
        value = JSONSchema.get_value(value,types)
            
        if section_name not in sections:
            sections[section_name] = OrderedDict([(JSONSchema.NAME,section_name)])
            
        if 'properties' not in sections[section_name]:
            sections[section_name]['properties'] = OrderedDict()
        
        # name should come first
        if JSONSchema.NAME == property_name and property_name not in sections[section_name]['properties']:
            new_dict = OrderedDict([(property_name, value)])
            new_dict.update(sections[section_name]['properties'])
            sections[section_name]['properties'] = new_dict
        else:
            sections[section_name]['properties'][property_name] = value
        
        # rebuild data
        contents = ''
        properties = sections[section_name]['properties']
        lastIndex = len(properties) - 1
        for i,(key,value) in enumerate(properties.items()):
            contents += '{}{}{}'.format(key,InputParser._PROPVALUE_SEPARATOR,value)
            if i < lastIndex:
                contents += '\n'
            
        sections[section_name]['data'] = contents
          
    def delete_section_property(self, section_name, property_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_name = JSONSchema.format_property_name(property_name)
        rebuild_data = False
        if section_name in self._sections and \
            'properties' in self._sections[section_name] and \
            property_name in self._sections[section_name]['properties']:
            del self._sections[section_name]['properties'][property_name]
            rebuild_data = True
          
        if rebuild_data:
            contents = ''
            properties = self._sections[section_name]['properties']
            lastIndex = len(properties) - 1
            for i,(key,value) in enumerate(properties.items()):
                contents += '{}{}{}'.format(key,InputParser._PROPVALUE_SEPARATOR,value)
                if i < lastIndex:
                    contents += '\n'
                
            self._sections[section_name]['data'] = contents
            
    def delete_section_properties(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        if section_name in self._sections:
            del self._sections[section_name]
    
    def set_section_data(self, section_name, value):
        """
        Sets a section data.
        Args:
            section_name (str): the name of the section, case insensitive
            value : value to set
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        value = self._json_schema.check_section_value(section_name,value)
        self._sections[section_name] = OrderedDict([(JSONSchema.NAME,section_name)])
        self._sections[section_name]['data'] = value
        properties = OrderedDict()
        if value is not None:
            lines = str(value).splitlines()
            for line in lines:
                k,v = self._get_key_value(line)
                if k is not None and v is not None:
                    properties[k] = v
    
        self._sections[section_name]['properties'] = properties
                
    def delete_section_data(self, section_name):
        """
        Deletes a section data.
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = JSONSchema.format_section_name(section_name).lower()
        if section_name in self._sections:
            self._sections[section_name]['data'] = ''
            self._sections[section_name]['properties'] = OrderedDict()

    def get_section_names(self):
        """Return all the names of the sections."""
        return list(self._sections.keys())
    
    def is_substitution_allowed(self):
        auto_substitutions = self.get_property_default_value(JSONSchema.PROBLEM,InputParser.AUTO_SUBSTITUTIONS)
        auto_substitutions = self.get_section_property(JSONSchema.PROBLEM,InputParser.AUTO_SUBSTITUTIONS,auto_substitutions)
        if auto_substitutions is None:
            auto_substitutions = True
            
        return auto_substitutions
     
    def check_if_substitution_key(self,section_name,property_names):
        result = [(property_name,False) for property_name in property_names]
        if not self.is_substitution_allowed():
            return result
        
        section_name = JSONSchema.format_section_name(section_name).lower()
        property_names = [JSONSchema.format_property_name(property_name) for property_name in property_names]
        section_property_name = self.get_property_default_value(section_name,JSONSchema.NAME)
        section_property_name = self.get_section_property(section_name,JSONSchema.NAME,section_property_name)
        for key in self._substitutions.keys():
            key_items = key.split('.')
            if len(key_items) == 3 and \
                key_items[0] == section_name and \
                key_items[1] == section_property_name and \
                key_items[2] in property_names:
                result[property_names.index(key_items[2])] = (key_items[2],True)
                continue
        
        return result
        
    def process_substitutions(self,substitutions = None):
        if substitutions is not None and not isinstance(substitutions,dict):
            raise AquaChemistryError('Invalid substitution parameter: {}'.format(substitutions))
            
        if not self.is_substitution_allowed():
            return {}
        
        result = {}
        for key,value in self._substitutions.items():
            key_items = key.split('.')
            if len(key_items) != 3:
                raise AquaChemistryError('Invalid substitution key: {}'.format(key))
                
            name = self.get_property_default_value(key_items[0],JSONSchema.NAME)
            name = self.get_section_property(key_items[0],JSONSchema.NAME,name)
            if name != key_items[1]:
                continue
            
            value_set = False
            value_items = value.split('.')
            if len(value_items) == 3:
                name = self.get_section_property(value_items[0],JSONSchema.NAME)
                if name == value_items[1]:
                    v = self.get_property_default_value(value_items[0],value_items[2])
                    v = self.get_section_property(value_items[0],value_items[2],v)
                    if v is not None:
                        self.set_section_property(key_items[0],key_items[2],v)
                        result[key] = v
                        value_set = True
                        
            if value_set or substitutions is None:
                continue
            
            if value in substitutions:
                self.set_section_property(key_items[0],key_items[2],substitutions[value])
                result[key] = substitutions[value]
                
        return result
            
    def _process_line(self,section,line):
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
                raise AquaChemistryError('New section "{0}" starting before the end of previuos section "{1}"'.format(line, section[JSONSchema.NAME]))
            
            return OrderedDict([(JSONSchema.NAME,stripLine[1:].lower()), ('data',[])])
        
        if section is None:
            return section
            
        section['data'].append(line)
       
        return section
    
    def _process_section(self,section):
        contents = ''
        sep_pos = -len(os.linesep)
        lastIndex = len(section['data']) - 1
        for i,line in enumerate(section['data']):
            key,value = self._get_key_value(line)
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
            return (None,None)
        
        if pos > 0:
            stripLine = stripLine[:pos].strip()
            
        pos = stripLine.find(InputParser._PROPVALUE_SEPARATOR)
        if pos > 0:
            key =  stripLine[0:pos].strip()
            value = stripLine[pos+1:].strip()
            return (key,JSONSchema.get_value(value))
        
        return (None,None)
        
            
        
            

