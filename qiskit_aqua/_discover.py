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
Methods for pluggable objects discovery, registration, information
"""

import os
import pkgutil
import importlib
import inspect
from collections import namedtuple
from .quantumalgorithm import QuantumAlgorithm
from qiskit_aqua import AlgorithmError
from qiskit_aqua.preferences import Preferences
from qiskit_aqua.utils.optimizers import Optimizer
from qiskit_aqua.utils.variational_forms import VariationalForm
from qiskit_aqua.utils.initial_states import InitialState
from qiskit_aqua.utils.iqfts import IQFT
from qiskit_aqua.utils.oracles import Oracle
import logging
import sys

logger = logging.getLogger(__name__)

_PLUGGABLES = {
    'algorithm': QuantumAlgorithm,
    'optimizer': Optimizer,
    'variational_form': VariationalForm,
    'initial_state': InitialState,
    'iqft': IQFT,
    'oracle': Oracle
}

_NAMES_TO_EXCLUDE = [
    '__main__',
    '_discover_qconfig',
    '_discover',
    '_logging',
    'algomethods',
    'algorithmerror',
    'operator',
    'preferences',
    'quantumalgorithm',
    'algoutils',
    'jsonutils',
    'optimizer',
    'variational_form',
    'initial_state',
    'iqft',
    'oracle'
]

_FOLDERS_TO_EXCLUDE = ['__pycache__','input','ui','parser']

RegisteredPluggable = namedtuple('RegisteredPluggable', ['name', 'cls', 'configuration'])

_REGISTERED_PLUGGABLES = {}

_DISCOVERED = False

def refresh_pluggables():
    """
    Attempts to rediscover all pluggable modules
    """
    global _REGISTERED_PLUGGABLES
    _REGISTERED_PLUGGABLES = {}
    global _DISCOVERED
    _DISCOVERED = True
    discover_local_pluggables() 
    discover_preferences_pluggables()
    if logger.isEnabledFor(logging.DEBUG):
        for ptype in local_pluggables_types():
            logger.debug("Found: '{}' has pluggables {} ".format(ptype, local_pluggables(ptype)))

def _discover_on_demand():
    """
    Attempts to discover pluggable modules, if not already discovered
    """
    global _DISCOVERED
    if not _DISCOVERED:
        _DISCOVERED = True
        discover_local_pluggables() 
        discover_preferences_pluggables()
        if logger.isEnabledFor(logging.DEBUG):
            for ptype in local_pluggables_types():
                logger.debug("Found: '{}' has pluggables {} ".format(ptype, local_pluggables(ptype)))

        
def discover_preferences_pluggables():
    """
    Discovers the pluggable modules on the directory and subdirectories of the preferences package
    and attempts to register them. Pluggable modules should subclass Pluggable Base classes.
    """
    preferences = Preferences()
    packages = preferences.get_packages([])
    for package in packages:
        try:
            mod = importlib.import_module(package)
            if mod is not None:
                _discover_local_pluggables(os.path.dirname(mod.__file__),
                                           mod.__name__,
                                           names_to_exclude=['__main__'],
                                           folders_to_exclude= ['__pycache__'])
            else:
                # Ignore package that could not be initialized.
                logger.debug('Failed to import package {}'.format(package))
        except Exception as e:
            # Ignore package that could not be initialized.
            logger.debug('Failed to load package {} error {}'.format(package, str(e)))
            
def _discover_local_pluggables(directory,
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
                            if cls.__module__ == modspec.name:
                                for pluggable_type,c in _PLUGGABLES.items():
                                    if issubclass(cls, c):
                                        _register_pluggable(pluggable_type,cls)
                                        importlib.import_module(fullname)
                                        break
                        except Exception as e:
                            # Ignore pluggables that could not be initialized.
                            logger.debug('Failed to load {} error {}'.format(fullname, str(e)))
                    
                except Exception as e:
                    # Ignore pluggables that could not be initialized.
                    logger.debug('Failed to load {} error {}'.format(fullname, str(e)))

        for item  in os.listdir(directory):
            fullpath = os.path.join(directory,item)
            if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
                _discover_local_pluggables(fullpath,parentname + '.' + item,names_to_exclude,folders_to_exclude)

def discover_local_pluggables(directory=os.path.dirname(__file__),
                           parentname=os.path.splitext(__name__)[0],
                           names_to_exclude=_NAMES_TO_EXCLUDE,
                           folders_to_exclude=_FOLDERS_TO_EXCLUDE):
    """
    Discovers the pluggable modules on the directory and subdirectories of the current module
    and attempts to register them. Pluggable modules should subclass Pluggable Base classes.
    Args:
        directory (str, optional): Directory to search for pluggable. Defaults
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
        _discover_local_pluggables(directory,parentname)
    finally:
        sys.path = syspath_save

def register_pluggable(cls, configuration=None):
    """
    Registers a pluggable class
    Args:
        cls (object): Pluggable class.
        configuration (object, optional): Pluggable configuration
     Returns:
        name: pluggable name
    """
    _discover_on_demand()
    pluggable_type = None
    for type,c in _PLUGGABLES.items():
        if issubclass(cls, c):
            pluggable_type = type
            break

    if pluggable_type is None:
        raise AlgorithmError('Could not register class {} is not subclass of any known pluggable'.format(cls))

    return _register_pluggable(pluggable_type,cls,configuration)

def _register_pluggable(pluggable_type, cls, configuration=None):
    """
    Registers a pluggable class
    Args:
        pluggable_type(str): The pluggable type
        cls (object): Pluggable class.
        configuration (object, optional): Pluggable configuration
     Returns:
        name: pluggable name
    Raises:
        AlgorithmError: if the class is already registered or could not be registered
    """
    if pluggable_type not in _REGISTERED_PLUGGABLES:
        _REGISTERED_PLUGGABLES[pluggable_type] = {}

    # Verify that the pluggable is not already registered.
    registered_classes = _REGISTERED_PLUGGABLES[pluggable_type]
    if cls in [pluggable.cls for pluggable in registered_classes.values()]:
        raise AlgorithmError('Could not register class {} is already registered'.format(cls))

    try:
        pluggable_instance = cls(configuration=configuration)
    except Exception as err:
        raise AlgorithmError('Could not register puggable:{} could not be instantiated: {}'.format(cls, str(err)))

    # Verify that it has a minimal valid configuration.
    try:
        pluggable_name = pluggable_instance.configuration['name']
    except (LookupError, TypeError):
        raise AlgorithmError('Could not register pluggable: invalid configuration')

    if pluggable_name in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AlgorithmError('Could not register class {}. Name {} {} is already registered'.format(cls,
                             pluggable_name,_REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name] = RegisteredPluggable(pluggable_name, cls, pluggable_instance.configuration)
    return pluggable_name

def deregister_pluggable(pluggable_type,pluggable_name):
    """
    Deregisters a pluggable class
    Args:
        pluggable_type(str): The pluggable type
        pluggable_name (str): The pluggable name
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AlgorithmError('Could not deregister {} {} not registered'.format(pluggable_type,pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AlgorithmError('Could not deregister {} {} not registered'.format(pluggable_type,pluggable_name))

    _REGISTERED_PLUGGABLES[pluggable_type].pop(pluggable_name)

def get_pluggable_class(pluggable_type,pluggable_name):
    """
    Accesses pluggable class
    Args:
        pluggable_type(str): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        cls: pluggable class
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    return _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].cls

def get_pluggable_instance(pluggable_type,pluggable_name):
    """
    Instantiates a pluggable class
    Args:
        pluggable_type(str): The pluggable type
        pluggable_name (str): The pluggable name
     Returns:
        instance: pluggable instance
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    return _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].cls(
            configuration=_REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].configuration)

def get_pluggable_configuration(pluggable_type,pluggable_name):
    """
    Accesses pluggable configuration
    Args:
        pluggable_type(str): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        configuration: pluggable configuration
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AlgorithmError('{} {} not registered'.format(pluggable_type,pluggable_name))

    return _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].configuration

def local_pluggables_types():
    """
    Accesses all pluggable types
    Returns:
       types: pluggable types
    """
    _discover_on_demand()
    return list(_REGISTERED_PLUGGABLES.keys())

def local_pluggables(pluggable_type):
    """
    Accesses pluggable names
    Args:
        pluggable_type(str): The pluggable type
    Returns:
        names: pluggable names
    Raises:
        AlgorithmError: if the tyoe is not registered
    """
    _discover_on_demand()
    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AlgorithmError('{} not registered'.format(pluggable_type))

    return [pluggable.name for pluggable in _REGISTERED_PLUGGABLES[pluggable_type].values()]

for pluggable_type in _PLUGGABLES.keys():
    method = 'def register_{}(cls, configuration=None): return register_pluggable(cls,configuration)'.format(pluggable_type)
    exec(method)
    method = "def deregister_{}(name): deregister_pluggable('{}',name)".format(pluggable_type,pluggable_type)
    exec(method)
    method = "def get_{}_class(name): return get_pluggable_class('{}',name)".format(pluggable_type,pluggable_type)
    exec(method)
    method = "def get_{}_instance(name): return get_pluggable_instance('{}',name)".format(pluggable_type,pluggable_type)
    exec(method)
    method = "def get_{}_configuration(name): return get_pluggable_configuration('{}',name)".format(pluggable_type,pluggable_type)
    exec(method)
    method = "def local_{}s(): return local_pluggables('{}')".format(pluggable_type,pluggable_type)
    exec(method)
