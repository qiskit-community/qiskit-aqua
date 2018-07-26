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
import ast
import json
import jsonschema
import os
from collections import OrderedDict
import logging
import copy
from qiskit_aqua import (local_pluggables_types,
                         get_pluggable_configuration,
                         get_algorithm_configuration,
                         local_algorithms)
from qiskit_aqua.input import (local_inputs, get_input_configuration)

logger = logging.getLogger(__name__)

class InputParser(object):
    """JSON input Parser."""
    
    NAME = 'name'
    INPUT = 'input'
    PROBLEM = 'problem'
    ALGORITHM = 'algorithm'
    BACKEND = 'backend'
   
    _UNKNOWN = 'unknown'
    _PROPERTY_ORDER = [NAME,_UNKNOWN]
    
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
                
        self._section_order = [InputParser.PROBLEM,InputParser.INPUT,InputParser.ALGORITHM]
        for pluggable_type in local_pluggables_types():
            if pluggable_type != InputParser.ALGORITHM:
                self._section_order.append(pluggable_type)
                
        self._section_order.extend([InputParser.BACKEND,InputParser._UNKNOWN])
        
        problems_dict = OrderedDict()
        for algo_name in local_algorithms():
            problems = InputParser.get_algorithm_problems(algo_name)
            for problem in problems:
                problems_dict[problem] = None
          
        problems_enum = { 'enum' : list(problems_dict.keys()) }
        jsonfile = os.path.join(os.path.dirname(__file__), 'input_schema.json')
        with open(jsonfile) as json_file:
            self._schema = json.load(json_file)
            self._schema['definitions'][InputParser.PROBLEM]['properties'][InputParser.NAME]['oneOf'] = [problems_enum]
            self._original_schema = copy.deepcopy(self._schema)
            
    def _order_sections(self,sections):
        sections_sorted = OrderedDict(sorted(list(sections.items()),
             key=lambda x: self._section_order.index(x[0]) 
             if x[0] in self._section_order else self._section_order.index(InputParser._UNKNOWN)))
        
        for section,properties in sections_sorted.items():
            if isinstance(properties,dict):
                sections_sorted[section] = OrderedDict(sorted(list(properties.items()),
                               key=lambda x: InputParser._PROPERTY_ORDER.index(x[0]) 
                               if x[0] in InputParser._PROPERTY_ORDER else InputParser._PROPERTY_ORDER.index(InputParser._UNKNOWN)))
            
        return sections_sorted
            
    def parse(self):
        """Parse the data."""
        if self._sections is None:
            if self._filename is None:
                raise AlgorithmError("Missing input file")
             
            with open(self._filename) as json_file:
                self._sections = json.load(json_file)
                
        self._update_pluggable_input_schemas()
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
        return InputParser._format_section_name(section_name) in local_pluggables_types()
        
    @staticmethod
    def _format_section_name(section_name):
        if section_name is None:
            section_name = ''
        section_name = section_name.strip()
        if len(section_name) == 0:
            raise AlgorithmError("Empty section name.")
            
        return section_name
    
    @staticmethod
    def _format_property_name(property_name):
        if property_name is None:
            property_name = ''
        property_name = property_name.strip()
        if len(property_name) == 0:
            raise AlgorithmError("Empty property name.")
            
        return property_name
    
    def get_section_types(self,section_name):
        section_name = InputParser._format_section_name(section_name)
        if 'definitions' not in self._schema:
            return []
        
        if section_name not in self._schema['definitions']:
            return []
        
        if 'type' not in self._schema['definitions'][section_name]:
            return []
        
        types = self._schema['definitions'][section_name]['type']
        if isinstance(types,list):
            return types
            
        return [types]
    
    def get_property_types(self,section_name,property_name):
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        if 'definitions' not in self._schema:
            return []
        
        if section_name not in self._schema['definitions']:
            return []
        
        if 'properties' not in self._schema['definitions'][section_name]:
            return []
        
        if property_name not in self._schema['definitions'][section_name]['properties']:
            return []
        
        prop = self._schema['definitions'][section_name]['properties'][property_name]
        if 'type' in prop:
            types  = prop['type']
            if isinstance(types,list):
                return types
            
            return [types]
            
        return []
    
    def get_default_sections(self):
        if 'definitions' not in self._schema:
            return None
        
        return copy.deepcopy(self._schema['definitions'])
    
    def get_default_section_names(self):
        sections = self.get_default_sections()
        return list(sections.keys()) if sections is not None else []
    
    def get_section_default_properties(self,section_name):
        section_name = InputParser._format_section_name(section_name)
        if 'definitions' not in self._schema:
            return None
        
        if section_name not in self._schema['definitions']:
            return None
        
        types = [self._schema['definitions'][section_name]['type']] if 'type' in self._schema['definitions'][section_name] else []
        
        if 'default' in self._schema['definitions'][section_name]:
            return InputParser._get_value(self._schema['definitions'][section_name]['default'],types)
        
        if 'object' not in types:
            return InputParser._get_value(None,types)
        
        if 'properties' not in self._schema['definitions'][section_name]:
            return None
        
        properties = OrderedDict()
        for property_name,values in self._schema['definitions'][section_name]['properties'].items():
            types = [values['type']] if 'type' in values else []
            default_value = values['default'] if 'default' in values else None
            properties[property_name] = InputParser._get_value(default_value,types)
           
        return properties
        
    def allows_additional_properties(self,section_name):
        section_name = InputParser._format_section_name(section_name)
        if 'definitions' not in self._schema:
            return True
        
        if section_name not in self._schema['definitions']:
            return True
        
        if 'additionalProperties' not in self._schema['definitions'][section_name]:
            return True
        
        return InputParser._get_value(self._schema['definitions'][section_name]['additionalProperties'])
        
    def get_property_default_values(self,section_name,property_name):
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        if 'definitions' not in self._schema:
            return None
        
        if section_name not in self._schema['definitions']:
            return None
        
        if 'properties' not in self._schema['definitions'][section_name]:
            return None
        
        if property_name not in self._schema['definitions'][section_name]['properties']:
            return None
        
        prop = self._schema['definitions'][section_name]['properties'][property_name]
        if 'type' in prop:
            types = prop['type']
            if not isinstance(types,list):
                types = [types]
                
            if 'boolean' in types:
                return [True,False]
            
        if 'oneOf' not in prop:
            return None
        
        for item in prop['oneOf']:
            if 'enum' in item:
                return item['enum']
            
        return None
    
    def get_property_default_value(self,section_name,property_name):
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        if 'definitions' not in self._schema:
            return None
        
        if section_name not in self._schema['definitions']:
            return None
        
        if 'properties' not in self._schema['definitions'][section_name]:
            return None
        
        if property_name not in self._schema['definitions'][section_name]['properties']:
            return None
        
        prop = self._schema['definitions'][section_name]['properties'][property_name]
        if 'default' in prop:
            return InputParser._get_value(prop['default'])
        
        return None
    
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
        config = get_algorithm_configuration(algo_name)
        if 'problems' in config:
            return config['problems']
        
        return []
                
    def _update_pluggable_input_schemas(self):
        # find alogorithm
        default_algo_name = self.get_property_default_value(InputParser.ALGORITHM,InputParser.NAME)
        algo_name = self.get_section_property(InputParser.ALGORITHM,InputParser.NAME,default_algo_name)
           
        # update alogorithm scheme
        if algo_name is not None:
            self._update_pluggable_input_schema(InputParser.ALGORITHM,algo_name,default_algo_name)
            
        # update alogorithm depoendencies scheme
        config = {} if algo_name is None else get_algorithm_configuration(algo_name) 
        classical = config['classical'] if 'classical' in config else False 
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {} if 'defaults' not in config else config['defaults']
        pluggable_types = local_pluggables_types()
        for pluggable_type in pluggable_types:
            if pluggable_type != InputParser.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from schema that ate not in the dependencies
                if pluggable_type in self._schema['definitions']:
                    del self._schema['definitions'][pluggable_type]
                if pluggable_type in self._schema['properties']:
                    del self._schema['properties'][pluggable_type] 
                    
        # update algorithm backend from schema if it is classical or not
        if classical:
            if InputParser.BACKEND in self._schema['definitions']:
                del self._schema['definitions'][InputParser.BACKEND]
            if InputParser.BACKEND in self._schema['properties']:
                del self._schema['properties'][InputParser.BACKEND]
        else:
            if InputParser.BACKEND not in self._schema['definitions']:
                self._schema['definitions'][InputParser.BACKEND] = self._original_schema['definitions'][InputParser.BACKEND]
            if InputParser.BACKEND not in self._schema['properties']:
                self._schema['properties'][InputParser.BACKEND] = self._original_schema['properties'][InputParser.BACKEND]
        
        # update schema with dependencies
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            default_properties = {}
            if pluggable_type in pluggable_defaults:
                for key,value in pluggable_defaults[pluggable_type].items():
                    if key == InputParser.NAME:
                        pluggable_name = pluggable_defaults[pluggable_type][key]
                    else:
                        default_properties[key] = value
              
            default_name = pluggable_name
            pluggable_name = self.get_section_property(pluggable_type,InputParser.NAME,pluggable_name)
            if pluggable_name is None:
                continue
            
            # update dependency schema
            self._update_pluggable_input_schema(pluggable_type,pluggable_name,default_name) 
            for property_name in self._schema['definitions'][pluggable_type]['properties'].keys():
                if property_name in default_properties:
                    self._schema['definitions'][pluggable_type]['properties'][property_name]['default'] = default_properties[property_name]
   
    def _update_algorithm_input_schema(self):
        # find alogorithm input
        default_name = self.get_property_default_value(InputParser.INPUT,InputParser.NAME)
        input_name = self.get_section_property(InputParser.INPUT,InputParser.NAME,default_name)
        if input_name is None:
            # find the first valid input for the problem
            problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
            if problem_name is None:
                raise AlgorithmError("No algorithm 'problem' section found on input.")
        
            for name in local_inputs():
                if problem_name in self.get_input_problems(name):
                    # set to the first input to solve the problem
                    input_name = name
                    break
                
        if input_name is None:
            # just remove fromm schema if none solves the problem
            if InputParser.INPUT in self._schema['definitions']:
                del self._schema['definitions'][InputParser.INPUT]
            if InputParser.INPUT in self._schema['properties']:
                del self._schema['properties'][InputParser.INPUT]
            return
        
        if default_name is None:
            default_name = input_name
        
        config = {}
        try:
            config = get_input_configuration(input_name)
        except:
            pass
            
        input_schema = config['input_schema'] if 'input_schema' in config else {}
        properties = input_schema['properties'] if 'properties' in input_schema else {}
        properties[InputParser.NAME] = { 'type': 'string' }
        required = input_schema['required'] if 'required' in input_schema else [] 
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True 
        if default_name is not None:
            properties[InputParser.NAME]['default'] = default_name
            required.append(InputParser.NAME) 
        
        if InputParser.INPUT not in self._schema['definitions']:
            self._schema['definitions'][InputParser.INPUT] = { 'type': 'object' }
            
        if InputParser.INPUT not in self._schema['properties']:
            self._schema['properties'][InputParser.INPUT] = {
                    '$ref': "#/definitions/{}".format(InputParser.INPUT)
            }
        
        self._schema['definitions'][InputParser.INPUT]['properties'] = properties
        self._schema['definitions'][InputParser.INPUT]['required'] = required
        self._schema['definitions'][InputParser.INPUT]['additionalProperties'] = additionalProperties
        
    def _update_pluggable_input_schema(self,pluggable_type,pluggable_name,default_name):
        config = {}
        try:
            config = get_pluggable_configuration(pluggable_type,pluggable_name)
        except:
            pass
            
        input_schema = config['input_schema'] if 'input_schema' in config else {}
        properties = input_schema['properties'] if 'properties' in input_schema else {}
        properties[InputParser.NAME] = { 'type': 'string' }
        required = input_schema['required'] if 'required' in input_schema else [] 
        additionalProperties = input_schema['additionalProperties'] if 'additionalProperties' in input_schema else True 
        if default_name is not None:
            properties[InputParser.NAME]['default'] = default_name
            required.append(InputParser.NAME) 
        
        if pluggable_type not in self._schema['definitions']:
            self._schema['definitions'][pluggable_type] = { 'type': 'object' }
            
        if pluggable_type not in self._schema['properties']:
            self._schema['properties'][pluggable_type] = {
                    '$ref': "#/definitions/{}".format(pluggable_type)
            }
        
        self._schema['definitions'][pluggable_type]['properties'] = properties
        self._schema['definitions'][pluggable_type]['required'] = required
        self._schema['definitions'][pluggable_type]['additionalProperties'] = additionalProperties
        
    def _merge_dependencies(self):
        algo_name = self.get_section_property(InputParser.ALGORITHM,InputParser.NAME)
        if algo_name is None:
            return
        
        config = get_algorithm_configuration(algo_name)
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {} if 'defaults' not in config else config['defaults']
        for pluggable_type in local_pluggables_types():
            if pluggable_type != InputParser.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from input that are not in the dependencies
                if pluggable_type in self._sections:
                   del self._sections[pluggable_type] 
        
        section_names = self.get_section_names()
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            new_properties = {}
            if pluggable_type in pluggable_defaults:
                for key,value in pluggable_defaults[pluggable_type].items():
                    if key == InputParser.NAME:
                        pluggable_name = pluggable_defaults[pluggable_type][key]
                    else:
                        new_properties[key] = value
                        
            if pluggable_name is None:
                continue
            
            if pluggable_type not in section_names:
                self.set_section(pluggable_type)
                
            if self.get_section_property(pluggable_type,InputParser.NAME) is None:
                self.set_section_property(pluggable_type,InputParser.NAME,pluggable_name)
               
            if pluggable_name == self.get_section_property(pluggable_type,InputParser.NAME):
                properties = self.get_section_properties(pluggable_type)
                if new_properties:
                    new_properties.update(properties)
                else:
                    new_properties = properties
                    
                self.set_section_properties(pluggable_type,new_properties)
                
    def _merge_default_values(self):
        section_names = self.get_section_names()
        if InputParser.ALGORITHM in section_names:
            if InputParser.PROBLEM not in section_names:
                self.set_section(InputParser.PROBLEM)
                
        self._update_pluggable_input_schemas()
        self._update_algorithm_input_schema()
        self._merge_dependencies()
      
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
        try:
            self._merge_default_values()
            json_dict = self.get_sections() 
            logger.debug('Algorithm Input: {}'.format(json.dumps(json_dict, sort_keys=True, indent=4)))
            logger.debug('Algorithm Input Schema: {}'.format(json.dumps(self._schema, sort_keys=True, indent=4)))
            jsonschema.validate(json_dict,self._schema)
        except jsonschema.exceptions.ValidationError as ve:
            logger.info('JSON Validation error: {}'.format(str(ve)))
            raise AlgorithmError(ve.message)
            
        self._validate_algorithm_problem()
        self._validate_input_problem()
            
    def _validate_algorithm_problem(self):
        algo_name = self.get_section_property(InputParser.ALGORITHM,InputParser.NAME)
        if algo_name is None:
            return
        
        problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
        if problem_name is None:
            raise AlgorithmError("No algorithm 'problem' section found on input.")
       
        problems = InputParser.get_algorithm_problems(algo_name)
        if problem_name not in problems:
            raise AlgorithmError(
            "Problem: {} not in the list of problems: {} for algorithm: {}.".format(problem_name,problems,algo_name))
    
    def _validate_input_problem(self):
        input_name = self.get_section_property(InputParser.INPUT,InputParser.NAME)
        if input_name is None:
            return
        
        problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
        if problem_name is None:
            raise AlgorithmError("No algorithm 'problem' section found on input.")
       
        problems = InputParser.get_input_problems(input_name)
        if problem_name not in problems:
            raise AlgorithmError(
            "Problem: {} not in the list of problems: {} for input: {}.".format(problem_name,problems,input_name))
           
    def commit_changes(self):
        self._original_sections = copy.deepcopy(self._sections)
        
    def save_to_file(self,file_name):
        if file_name is None:
            raise AlgorithmError('Missing file path')
            
        file_name = file_name.strip()
        if len(file_name) == 0:
            raise AlgorithmError('Missing file path')
            
        with open(file_name, 'w') as f:
            print(json.dumps(self.get_sections(), sort_keys=True, indent=4), file=f)            
               
    def section_is_text(self,section_name):
        section_name = InputParser._format_section_name(section_name)
        types = self.get_section_types(section_name)
        if len(types) > 0: 
            return 'object' not in types
        
        section = self.get_section(section_name)
        if section is None:
            return False
        
        return not isinstance(section,dict)
    
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
        section_name = InputParser._format_section_name(section_name)     
        try:
            return self._sections[section_name]
        except KeyError:
            raise AlgorithmError('No section "{0}"'.format(section_name))
            
    def get_section_text(self,section_name):
        section = self.get_section(section_name)
        if section is None:
            return ''
        
        if isinstance(section,str):
            return str
        
        return json.dumps(section, sort_keys=True, indent=4)
     
    def get_section_properties(self,section_name):
        section = self.get_section(section_name)
        if section is None:
            return {}
       
        return section
           
    def get_section_property(self, section_name, property_name, default_value = None):
        """Return a property by name.
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            default_value : default value in case it is not found
        Returns:
            Value: The property value
        """
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
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
        section_name = InputParser._format_section_name(section_name)
        if section_name not in self._sections:
            self._sections[section_name] = OrderedDict()
            self._sections = self._order_sections(self._sections)
            
    def delete_section(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = InputParser._format_section_name(section_name)
        if section_name not in self._sections:
            return
            
        del self._sections[section_name]
            
        # update schema
        self._schema = copy.deepcopy(self._original_schema)
        self._update_pluggable_input_schemas()
        self._update_algorithm_input_schema()
           
    def set_section_properties(self, section_name, properties):
        self.delete_section_properties(section_name)
        for property_name,value in properties.items():
            self.set_section_property(section_name,property_name,value)
    
    def set_section_property(self, section_name, property_name, value):
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        types = self.get_property_types(section_name,property_name)
        value = InputParser._get_value(value,types)
        if len(types) > 0:
            validator = jsonschema.Draft4Validator(self._schema)
            valid = False
            for type in types:
                valid = validator.is_type(value,type)
                if valid:
                    break
            
            if not valid:
                raise AlgorithmError("{}.{}: Value '{}' is not of types: '{}'".format(section_name,property_name,value,types)) 
            
        sections_temp = copy.deepcopy(self._sections)
        InputParser._set_section_property(sections_temp,section_name,property_name,value,types)
        msg = self._validate(sections_temp,section_name, property_name)
        if msg is not None:
            raise AlgorithmError("{}.{}: Value '{}': '{}'".format(section_name,property_name,value,msg)) 
      
        InputParser._set_section_property(self._sections,section_name,property_name,value,types)
        if property_name == InputParser.NAME:
            if InputParser.INPUT == section_name:
                self._update_algorithm_input_schema()
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties,dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != InputParser.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name,property_name)
            elif InputParser.PROBLEM == section_name:
                self._update_algorithm_problem()
                self._update_input_problem()
            elif InputParser.is_pluggable_section(section_name):
                self._update_pluggable_input_schemas()
                # remove properties that are not valid for this section
                default_properties = self.get_section_default_properties(section_name)
                if isinstance(default_properties,dict):
                    properties = self.get_section_properties(section_name)
                    for property_name in list(properties.keys()):
                        if property_name != InputParser.NAME and property_name not in default_properties:
                            self.delete_section_property(section_name,property_name)
                
                if section_name == InputParser.ALGORITHM:
                    self._update_dependency_sections()
                    
        self._sections = self._order_sections(self._sections)
       
    def _validate(self,sections,section_name, property_name):
        validator = jsonschema.Draft4Validator(self._schema)
        for error in sorted(validator.iter_errors(sections), key=str):
            if len(error.path) == 2 and error.path[0] == section_name and error.path[1] == property_name:
                return error.message
          
        return None
    
    def _update_algorithm_problem(self):
        problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
        if problem_name is None:
            raise AlgorithmError("No algorithm 'problem' section found on input.")
            
        algo_name = self.get_section_property(InputParser.ALGORITHM,InputParser.NAME)
        if algo_name is not None and problem_name in InputParser.get_algorithm_problems(algo_name):
            return
        
        for algo_name in local_algorithms():
            if problem_name in self.get_algorithm_problems(algo_name):
                # set to the first algorithm to solve the problem
                self.set_section_property(InputParser.ALGORITHM,InputParser.NAME,algo_name)
                return
            
        # no algorithm solve this problem, remove section
        self.delete_section(InputParser.ALGORITHM)
        
    def _update_input_problem(self):
        problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
        if problem_name is None:
            raise AlgorithmError("No algorithm 'problem' section found on input.")
            
        input_name = self.get_section_property(InputParser.INPUT,InputParser.NAME)
        if input_name is not None and problem_name in InputParser.get_input_problems(input_name):
            return
        
        for input_name in local_inputs():
            if problem_name in self.get_input_problems(input_name):
                # set to the first input to solve the problem
                self.set_section_property(InputParser.INPUT,InputParser.NAME,input_name)
                return
            
        # no input solve this problem, remove section
        self.delete_section(InputParser.INPUT)
        
    def _update_dependency_sections(self):
        algo_name = self.get_section_property(InputParser.ALGORITHM,InputParser.NAME)
        config = {} if algo_name is None else get_algorithm_configuration(algo_name) 
        classical = config['classical'] if 'classical' in config else False 
        pluggable_dependencies = [] if 'depends' not in config else config['depends']
        pluggable_defaults = {} if 'defaults' not in config else config['defaults']
        pluggable_types = local_pluggables_types()
        for pluggable_type in pluggable_types:
            if pluggable_type != InputParser.ALGORITHM and pluggable_type not in pluggable_dependencies:
                # remove pluggables from input that are not in the dependencies
                if pluggable_type in self._sections:
                   del self._sections[pluggable_type] 
       
        for pluggable_type in pluggable_dependencies:
            pluggable_name = None
            if pluggable_type in pluggable_defaults:
                if InputParser.NAME in pluggable_defaults[pluggable_type]:
                    pluggable_name = pluggable_defaults[pluggable_type][InputParser.NAME]
           
            if pluggable_name is not None and pluggable_type not in self._sections:
                self.set_section_property(pluggable_type,InputParser.NAME,pluggable_name)
               
        # update backend based on classical
        if classical:
            if InputParser.BACKEND in self._sections:
                del self._sections[InputParser.BACKEND]
        else:
            if InputParser.BACKEND not in self._sections:
                self._sections[InputParser.BACKEND] = self.get_section_default_properties(InputParser.BACKEND)
                    
    @staticmethod
    def _set_section_property(sections, section_name, property_name, value, types):
        """
        Args:
            section_name (str): the name of the section, case insensitive
            property_name (str): the property name in the section
            value : property value
            types : schema types
        """
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        value = InputParser._get_value(value,types)
            
        if section_name not in sections:
            sections[section_name] = OrderedDict()
       
        # name should come first
        if InputParser.NAME == property_name and property_name not in sections[section_name]:
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
        section_name = InputParser._format_section_name(section_name)
        property_name = InputParser._format_property_name(property_name)
        if section_name in self._sections and property_name in self._sections[section_name]:
            del self._sections[section_name][property_name]
            
    def delete_section_properties(self, section_name):
        """
        Args:
            section_name (str): the name of the section, case insensitive
        """
        section_name = InputParser._format_section_name(section_name)
        if section_name in self._sections:
            del self._sections[section_name]
            
    def set_section_data(self, section_name, value):
        """
        Sets a section data.
        Args:
            section_name (str): the name of the section, case insensitive
            value : value to set
        """
        section_name = InputParser._format_section_name(section_name)    
        types = self.get_section_types(section_name)
        value = InputParser._get_value(value,types)
        if len(types) > 0:
            validator = jsonschema.Draft4Validator(self._schema)
            valid = False
            for type in types:
                valid = validator.is_type(value,type)
                if valid:
                    break
            
            if not valid:
                raise AlgorithmError("{}: Value '{}' is not of types: '{}'".format(section_name,value,types)) 
        
        self._sections[section_name] = value

    def get_section_names(self):
        """Return all the names of the sections."""
        return list(self._sections.keys())
        
    @staticmethod
    def _get_value(value, types=[]):
        if value is None or (isinstance(value,str) and len(value.strip()) == 0):
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
            str_value = str(value)
            if str_value.lower() == 'true':
                return True
            elif str_value.lower() == 'false':
                return False
            
            v = ast.literal_eval(str_value)
            if isinstance(v,dict):
                v = json.loads(json.dumps(v))
            
            return v
        except:
            return value
