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

"""
Methods for chemistry operators objects discovery, registration, information
"""

import os
import pkgutil
import importlib
import inspect
from collections import namedtuple
from .chemistry_operator import ChemistryOperator
from qiskit_acqua_chemistry import ACQUAChemistryError
from qiskit_acqua_chemistry.preferences import Preferences
import logging
import sys

logger = logging.getLogger(__name__)

_NAMES_TO_EXCLUDE = ['_discover_chemoperator']

_FOLDERS_TO_EXCLUDE = ['__pycache__']

RegisteredChemOp = namedtuple('RegisteredChemOp', ['name', 'cls', 'configuration'])

_REGISTERED_CHEMISTRY_OPERATORS = {}

_DISCOVERED = False

def refresh_operators():
    """
    Attempts to rediscover all operator modules
    """
    global _REGISTERED_CHEMISTRY_OPERATORS
    _REGISTERED_CHEMISTRY_OPERATORS = {}
    global _DISCOVERED
    _DISCOVERED = True
    discover_local_chemistry_operators() 
    discover_preferences_chemistry_operators()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Found: chemistry operators {} ".format(local_chemistry_operators()))

def _discover_on_demand():
    """
    Attempts to discover operator modules, if not already discovered
    """
    global _DISCOVERED
    if not _DISCOVERED:
        _DISCOVERED = True
        discover_local_chemistry_operators() 
        discover_preferences_chemistry_operators()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Found: chemistry operators {} ".format(local_chemistry_operators()))

def discover_preferences_chemistry_operators():
    """
    Discovers the chemistry operators on the directory and subdirectories of the preferences package
    and attempts to register them. Chem.Operator modules should subclass ChemistryOperator Base class.
    """
    preferences = Preferences()
    packages = preferences.get_packages(Preferences.PACKAGE_TYPE_CHEMISTRY,[])
    for package in packages:
        try:
            mod = importlib.import_module(package)
            if mod is not None:
                _discover_local_chemistry_operators(os.path.dirname(mod.__file__),
                                           mod.__name__,
                                           names_to_exclude=['__main__'],
                                           folders_to_exclude= ['__pycache__'])
            else:
                # Ignore package that could not be initialized.
                logger.debug('Failed to import package {}'.format(package))
        except Exception as e:
            # Ignore package that could not be initialized.
            logger.debug('Failed to load package {} error {}'.format(package, str(e)))
        
def _discover_local_chemistry_operators(directory,
                                        parentname,
                                        names_to_exclude=_NAMES_TO_EXCLUDE,
                                        folders_to_exclude=_FOLDERS_TO_EXCLUDE):
        for _, name, ispackage in pkgutil.iter_modules([directory]):
            if ispackage:
                continue
            
            # Iterate through the modules
            if name not in names_to_exclude:  # skip those modules
                try:
                    fullname = parentname + '.' + name
                    modspec = importlib.util.find_spec(fullname)
                    mod = importlib.util.module_from_spec(modspec)
                    modspec.loader.exec_module(mod)
                    for _, cls in inspect.getmembers(mod, inspect.isclass):
                        # Iterate through the classes defined on the module.
                        try:
                            if cls.__module__ == modspec.name and issubclass(cls, ChemistryOperator):
                                register_chemistry_operator(cls)
                                importlib.import_module(fullname)
                        except Exception as e:
                            # Ignore operator that could not be initialized.
                            logger.debug('Failed to load {} error {}'.format(fullname, str(e)))
                except Exception as e:
                    # Ignore operator that could not be initialized.
                    logger.debug('Failed to load {} error {}'.format(fullname, str(e)))
                    
        for item  in os.listdir(directory):
            fullpath = os.path.join(directory,item)
            if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
                _discover_local_chemistry_operators(fullpath,parentname + '.' + item,names_to_exclude,folders_to_exclude)

def discover_local_chemistry_operators(directory=os.path.dirname(__file__),
                           parentname=os.path.splitext(__name__)[0]):
    """
    Discovers the chemistry operators modules on the directory and subdirectories of the current module
    and attempts to register them. Chem.Operator modules should subclass ChemistryOperator Base class.
    Args:
        directory (str, optional): Directory to search for input modules. Defaults
            to the directory of this module.
        parentname (str, optional): Module parent name. Defaults to current directory name
    """
    
    def _get_sys_path(directory):  
        syspath = [os.path.abspath(directory)]
        for item  in os.listdir(directory):
            fullpath = os.path.join(directory,item)
            if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                syspath += _get_sys_path(fullpath)
                
        return syspath
    
    syspath_save = sys.path
    sys.path = _get_sys_path(directory) + sys.path
    try:
        _discover_local_chemistry_operators(directory,parentname)
    finally:
        sys.path = syspath_save

def register_chemistry_operator(cls, configuration=None):
    """
    Registers a chemistry operator class
    Args:
        cls (object): chemistry operator class.
        configuration (object, optional): Pluggable configuration
    Returns:
        name: input name
    Raises:
        ACQUAChemistryError: if the class is already registered or could not be registered
    """
    _discover_on_demand()
       
    # Verify that the pluggable is not already registered 
    if cls in [input.cls for input in _REGISTERED_CHEMISTRY_OPERATORS.values()]:
        raise ACQUAChemistryError('Could not register class {} is already registered'.format(cls))

    try:
        chem_instance = cls(configuration=configuration)
    except Exception as err:
        raise ACQUAChemistryError('Could not register chemistry operator:{} could not be instantiated: {}'.format(cls, str(err)))

    # Verify that it has a minimal valid configuration.
    try:
        chemistry_operator_name = chem_instance.configuration['name']
    except (LookupError, TypeError):
        raise ACQUAChemistryError('Could not register chemistry operator: invalid configuration')
        
    if chemistry_operator_name in _REGISTERED_CHEMISTRY_OPERATORS:
        raise ACQUAChemistryError('Could not register class {}. Name {} {} is already registered'.format(cls,
                                                                                                         chemistry_operator_name, _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name] = RegisteredChemOp(chemistry_operator_name, cls, chem_instance.configuration)
    return chemistry_operator_name

def deregister_chemistry_operator(chemistry_operator_name):
    """
    Deregisters a chemistry operator class
    Args:
        chemistry_operator_name(str): The chemistry operator name
    Raises:
        ACQUAChemistryError: if the class is not registered
    """
    _discover_on_demand()
  
    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise ACQUAChemistryError('Could not deregister {} not registered'.format(chemistry_operator_name))
            
    _REGISTERED_CHEMISTRY_OPERATORS.pop(chemistry_operator_name)
    
def get_chemistry_operator_class(chemistry_operator_name):
    """
    Accesses chemistry operator class
    Args:
        chemistry_operator_name (str): The chemistry operator name
    Returns:
        cls: chemistry operator class
    Raises:
        ACQUAChemistryError: if the class is not registered
    """
    _discover_on_demand()
  
    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise ACQUAChemistryError('{} not registered'.format(chemistry_operator_name))
        
    return _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].cls

def get_chemistry_operator_instance(chemistry_operator_name):
    """
    Instantiates a chemistry operator class
    Args:
        chemistry_operator_name (str): The chemistry operator name 
    Returns:
        instance: chemistry operator instance
    Raises:
        ACQUAChemistryError: if the class is not registered
    """
    _discover_on_demand()
  
    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise ACQUAChemistryError('{} not registered'.format(chemistry_operator_name))
           
    return _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].cls(configuration=_REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].configuration)
    
def get_chemistry_operator_configuration(chemistry_operator_name):
    """
    Accesses chemistry operator configuration
    Args:
        chemistry_operator_name (str): The chemistry operator name
    Returns:
        configuration: chemistry operator configuration
    Raises:
        ACQUAChemistryError: if the class is not registered
    """
    _discover_on_demand()
  
    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise ACQUAChemistryError('{} not registered'.format(chemistry_operator_name))
        
    return _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].configuration

def local_chemistry_operators():
    """
    Accesses chemistry operator names
    Returns:
        names: chemistry operator names
    """
    _discover_on_demand()
    return [input.name for input in _REGISTERED_CHEMISTRY_OPERATORS.values()]
