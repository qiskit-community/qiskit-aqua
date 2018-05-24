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
import logging
import json
import jsonschema
from collections import OrderedDict
import sys
import importlib
import inspect
import copy
from ._basedriver import BaseDriver

logger = logging.getLogger(__name__)

"""Singleton configuration class."""
 
class ConfigurationManager(object):
    
    __INSTANCE = None # Shared instance
        
    def __init__(self):
        """ Create singleton instance """
        if ConfigurationManager.__INSTANCE is None:
            ConfigurationManager.__INSTANCE = ConfigurationManager.__ConfigurationManager()

        # Store instance reference as the only member in the handle
        self.__dict__['_ConfigurationManager__instance'] = ConfigurationManager.__INSTANCE

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__INSTANCE, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__INSTANCE, attr, value)
    
    
    class __ConfigurationManager(object):
        
        __CONFIGURATION_FILE = 'configuration.json'
        __CONFIGURATION_SCHEMA = 'configuration_schema.json'
        
        def __init__(self):
            self._discovered = False
            self._registration = OrderedDict()
            jsonfile = os.path.join(os.path.dirname(__file__), 
                                    ConfigurationManager.__ConfigurationManager.__CONFIGURATION_SCHEMA)
            with open(jsonfile) as json_file:
                self.schema = json.load(json_file)
                
        def register_driver(self,cls, configuration):
            """Register a driver.
            Register a class driver validating that:
            * it follows the `BaseDriver` specification.
            * it can instantiated in the current context.
            * the driver is not already registered.
            Args:
                cls (class): a subclass of BaseDriver 
                configuration (dict): driver configuration
            Returns:
                string: the identifier of the driver
            Raises:
                LookupError: if `cls`or configuration are not valid or already registered
            """
            try:
                jsonschema.validate(configuration,self.schema)
            except Exception as err:
                raise LookupError('Could not register driver: invalid configuration: {}'.format(err))
            
            if not issubclass(cls, BaseDriver):
                raise LookupError('Could not register driver: {} is not a subclass of BaseDriver'.format(cls))
                
            self._discover_on_demand()
            # Verify that the driver is not already registered.
            name = configuration['name']
            if name in self._registration:
                raise LookupError('Could not register driver: {}. Already registered.'.format(name))
            
            self._registration[name] = { 
                        'path': None,
                        'fullname': None,
                        'configuration':configuration,
                        'class': cls
                    }
            return name
    
        def deregister_driver(self,name):
            """Remove driver from list of available drivers
            Args:
                name (str): name of driver to unregister
            Raises:
                KeyError if name is not registered.
            """
            self._discover_on_demand()
            self._registration.pop(name)
                
        def get_driver_class(self,name):
            """Return the class object for the named module.
            Args:
                name (str): the module name
            Returns:
                Clas: class object for module
            Raises:
                LookupError: if module is unavailable
            """
            self._discover_on_demand()
            try:
                registered_module = self._registration[name]
                if registered_module['class'] is None:
                    registered_module['class'] = self._load_module(registered_module)
                
                return registered_module['class']
            except KeyError:
                raise LookupError('Driver "{}" is not available'.format(name))
                
        def get_driver_configuration(self,name):
            """Return the configuration for the named module.
            Args:
                name (str): the module name
            Returns:
                dict: configuration dict
            Raises:
                LookupError: if module is unavailable
            """
            self._discover_on_demand()
            try:
                return self._registration[name]['configuration']
            except KeyError:
                raise LookupError('Driver "{}" is not available'.format(name))
    
        def get_driver_instance(self,name):
            """Return an instance for the name in configuration.
            Args:
                name (str): the name
            Returns:
                Object: module instance 
            Raises:
                LookupError: if module is unavailable
            """
            cls = self.get_driver_class(name)
            config = self.get_driver_configuration(name)
            try:
                return cls(configuration=config)
            except Exception as err:
                raise LookupError('{} could not be instantiated: {}'.format(cls, err))
    
        @property
        def configurations(self):
            """Return configurations"""
            self._discover_on_demand()
            configurations = OrderedDict()
            for name,value in self._registration.items():
                configurations[name] = copy.deepcopy(value['configuration'])
            
            return configurations
        
        @property
        def module_names(self):
            """Return names"""
            self._discover_on_demand()
            return list(self._registration.keys())
             
        def _discover_on_demand(self):
            if not self._discovered:
                self._registration = OrderedDict()
                self.discover_configurations(os.path.dirname(__file__),
                                             os.path.splitext(__name__)[0])
                self._discovered = True
         
        def discover_configurations(self,directory,parentname):
            """
            This function looks for configuration.json files and attempts to load it
            Args:
                directory (str): Directory to search. 
                parentname: (str) parent module name
            Returns:
                exception list
            """
            directory = os.path.abspath(directory)
            for item in os.listdir(directory):
                fullpath = os.path.join(directory,item)
                if item.lower() == ConfigurationManager.__ConfigurationManager.__CONFIGURATION_FILE:
                    with open(fullpath) as json_file:
                        try:
                            json_dict = json.load(json_file)
                            jsonschema.validate(json_dict,self.schema)
                            module = json_dict['module']
                            if not os.path.isfile(os.path.join(directory,module + '.py')):
                                raise LookupError('Module {} not found.'.format(module))
                            
                            self._registration[json_dict['name']] = { 
                                        'path': directory,
                                        'fullname': parentname + '.' + module,
                                        'configuration':json_dict,
                                        'class': None
                                    }
                        except Exception as e:
                            logger.info('Configuration error: {}'.format(str(e)))
                            raise
                        
                    continue
                
                if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                    self.discover_configurations(fullpath,parentname + '.' + item)
        
        @staticmethod
        def _get_sys_path(directory):  
            syspath = [os.path.abspath(directory)]
            for item  in os.listdir(directory):
                fullpath = os.path.join(directory,item)
                if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                    syspath += ConfigurationManager.__ConfigurationManager._get_sys_path(fullpath)
                    
            return syspath
        
        def _load_module(self,registered_module):
            """This function attempts to load the registered module.
                Args:
                    registered module
                Returns:
                    module class
            """
            module_class = None
            syspath_save = sys.path
            sys.path = ConfigurationManager.__ConfigurationManager._get_sys_path(registered_module['path']) + sys.path
            try:
                modspec = importlib.util.find_spec(registered_module['fullname'])
                mod = importlib.util.module_from_spec(modspec)
                modspec.loader.exec_module(mod)
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    # Iterate through the classes defined on the module.
                    if (issubclass(cls, BaseDriver) and cls.__module__ == modspec.name):
                        module_class = cls
                        break
            finally:
                sys.path = syspath_save
            
            return module_class
