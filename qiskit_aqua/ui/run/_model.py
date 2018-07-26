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

import os
import json
from qiskit_aqua import AlgorithmError
from qiskit_aqua.parser import InputParser
from qiskit_aqua import local_pluggables
from qiskit_aqua.input import local_inputs
from qiskit_aqua.ui._uipreferences import UIPreferences
from collections import OrderedDict

class Model(object):
   
    def __init__(self):
        """Create Model object."""
        self._parser = InputParser()
        
    def is_empty(self):
        return self._parser is None or len(self._parser.get_section_names()) == 0
    
    def new(self):
        try:
            dict = {}
            jsonfile = os.path.join(os.path.dirname(__file__), 'input_template.json')
            with open(jsonfile) as json_file:
                dict = json.load(json_file)
                
            self._parser = InputParser(dict)
            self._parser.parse()
            uipreferences = UIPreferences()
            if uipreferences.get_populate_defaults(True):
                self._parser.validate_merge_defaults()
                self._parser.commit_changes()
                
            return self._parser.get_section_names()   
        except:
            self._parser = None
            raise
        
    def load_file(self,filename):
        if filename is None:
            return []
        try:
            self._parser = InputParser(filename)
            self._parser.parse()
            uipreferences = UIPreferences()
            if uipreferences.get_populate_defaults(True):
                self._parser.validate_merge_defaults()
                self._parser.commit_changes()
                
            return self._parser.get_section_names()   
        except:
            self._parser = None
            raise
            
    def get_filename(self):
        if self._parser is None:
            return None
    
        return self._parser.get_filename()
            
    def is_modified(self):
        if self._parser is None:
            return False
                
        return self._parser.is_modified()
        
    def save_to_file(self,filename):
        if self.is_empty():
            raise AlgorithmError("Empty input data.")
                
        self._parser.save_to_file(filename) 
            
    def get_section_names(self):
        if self._parser is None:
            return []
        
        return self._parser.get_section_names()
    
    def get_property_default_values(self,section_name,property_name):
        if self._parser is None:
            return None
        
        return self._parser.get_property_default_values(section_name,property_name)
    
    def section_is_text(self,section_name):
        if self._parser is None:
            return False
        
        return self._parser.section_is_text(section_name)
    
    def get_section(self,section_name):
        return self._parser.get_section(section_name) if self._parser is not None else None
    
    def get_section_text(self,section_name):
        if self._parser is None:
            return ''
        
        return self._parser.get_section_text(section_name)
     
    def get_section_properties(self,section_name):
        if self._parser is None:
            return {}
        
        return self._parser.get_section_properties(section_name)
    
    def default_properties_equals_properties(self,section_name):
        if self.section_is_text(section_name): 
            return self.get_section_default_properties(section_name) == self.get_section_data(section_name)
        
        default_properties = self.get_section_default_properties(section_name)
        if isinstance(default_properties,OrderedDict):
            default_properties =  dict(default_properties)
            
        properties = self.get_section_properties(section_name)
        if isinstance(properties,OrderedDict):
            properties =  dict(properties)
                
        if not isinstance(default_properties,dict) or not isinstance(properties,dict):
            return default_properties == properties
            
        if InputParser.NAME in properties:
            default_properties[InputParser.NAME] = properties[InputParser.NAME]
            
        return default_properties == properties
    
    def get_section_property(self,section_name,property_name):
        if self._parser is None:
            return None
        
        return self._parser.get_section_property(section_name,property_name)
    
    def set_section(self,section_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.set_section(section_name)
        value = self._parser.get_section_default_properties(section_name)
        if isinstance(value,dict):
            for property_name,property_value in value.items():
                self._parser.set_section_property(section_name,property_name,property_value)
                
            # do one more time in case schema was updated
            value = self._parser.get_section_default_properties(section_name)
            for property_name,property_value in value.items():
                self._parser.set_section_property(section_name,property_name,property_value)             
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
                
            self._parser.set_section_data(section_name,value)
            
    def set_default_properties_for_name(self,section_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        name = self._parser.get_section_property(section_name,InputParser.NAME)
        self._parser.delete_section_properties(section_name)
        value = self._parser.get_section_default_properties(section_name)
        if name is not None:
            self._parser.set_section_property(section_name,InputParser.NAME,name)
        if isinstance(value,dict):
            for property_name,property_value in value.items():
                if property_name != InputParser.NAME:
                    self._parser.set_section_property(section_name,property_name,property_value)
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
                
            self._parser.set_section_data(section_name,value)
            
    @staticmethod
    def is_pluggable_section(section_name):
        return InputParser.is_pluggable_section(section_name)
    
    def get_input_section_names(self):
        problem_name = None
        if self._parser is not None:
            problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
        if problem_name is None:
            problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                
        if problem_name is None:
            return local_inputs()
            
        input_names = []
        for input_name in local_inputs():
            problems = InputParser.get_input_problems(input_name)
            if problem_name in problems:
                input_names.append(input_name)
            
        return input_names
       
    def get_pluggable_section_names(self,section_name):
        if not Model.is_pluggable_section(section_name):
            return []
        
        if InputParser.ALGORITHM == section_name:
            problem_name = None
            if self._parser is not None:
                problem_name = self.get_section_property(InputParser.PROBLEM,InputParser.NAME)
            if problem_name is None:
                problem_name = self.get_property_default_value(InputParser.PROBLEM,InputParser.NAME)
                    
            if problem_name is None:
                return local_pluggables(InputParser.ALGORITHM)
           
            algo_names = []
            for algo_name in local_pluggables(InputParser.ALGORITHM):
                problems = InputParser.get_algorithm_problems(algo_name)
                if problem_name in problems:
                    algo_names.append(algo_name)
                
            return algo_names
        
        return local_pluggables(section_name)
             
    def delete_section(self, section_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.delete_section(section_name)
        
    def get_default_sections(self):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        return self._parser.get_default_sections()
         
    def get_section_default_properties(self,section_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        return self._parser.get_section_default_properties(section_name)
    
    def allows_additional_properties(self,section_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
        
        return self._parser.allows_additional_properties(section_name)
    
    def get_property_default_value(self,section_name,property_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        return self._parser.get_property_default_value(section_name,property_name)
    
    def get_property_types(self,section_name,property_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        return self._parser.get_property_types(section_name,property_name)
    
    def set_section_property(self, section_name, property_name, value):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.set_section_property(section_name,property_name,value)
        if property_name == InputParser.NAME and \
            (InputParser.is_pluggable_section(section_name) or section_name == InputParser.INPUT):
            properties = self._parser.get_section_default_properties(section_name)
            if isinstance(properties,dict):
                properties[ InputParser.NAME] = value
                self._parser.delete_section_properties(section_name)
                for property_name,property_value in properties.items():  
                    self._parser.set_section_property(section_name,property_name,property_value)
        
    def delete_section_property(self, section_name, property_name):
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.delete_section_property(section_name, property_name)
        if property_name == InputParser.NAME and \
            (InputParser.is_pluggable_section(section_name) or section_name == InputParser.INPUT):
            self._parser.delete_section_properties(section_name)
            
    def set_section_text(self, section_name, value):  
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.set_section_data(section_name, value)
        
    def delete_section_text(self, section_name):  
        if self._parser is None:
            raise AlgorithmError('Input not initialized.')
            
        self._parser.delete_section_text(section_name)
