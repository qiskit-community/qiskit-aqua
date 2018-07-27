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
Methods for input objects discovery, registration, information
"""

import os
import pkgutil
import importlib
import inspect
from collections import namedtuple
from qiskit_aqua.input import AlgorithmInput
from qiskit_aqua import AlgorithmError
import logging
import sys

logger = logging.getLogger(__name__)

_NAMES_TO_EXCLUDE = ['_discover_input',]

_FOLDERS_TO_EXCLUDE = ['__pycache__']

RegisteredInput = namedtuple('RegisteredInput', ['name', 'cls', 'configuration'])

_REGISTERED_INPUTS = {}

_DISCOVERED = False

def _discover_on_demand():
    """
    Attempts to discover input modules, if not already discovered
    """
    if not _DISCOVERED:
        discover_local_inputs()

def discover_local_inputs(directory=os.path.dirname(__file__),
                           parentname=os.path.splitext(__name__)[0]):
    """
    Discovers the input modules on the directory and subdirectories of the current module
    and attempts to register them. Pluggable modules should subclass AlgorithmInput Base class.
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

    def _discover_localinputs(directory,parentname):
        for _, name, ispackage in pkgutil.iter_modules([directory]):
            if ispackage:
                continue

            # Iterate through the modules
            if name not in _NAMES_TO_EXCLUDE:  # skip those modules
                try:
                    fullname = parentname + '.' + name
                    modspec = importlib.util.find_spec(fullname)
                    mod = importlib.util.module_from_spec(modspec)
                    modspec.loader.exec_module(mod)
                    for _, cls in inspect.getmembers(mod, inspect.isclass):
                        # Iterate through the classes defined on the module.
                        if cls.__module__ == modspec.name and issubclass(cls, AlgorithmInput):
                            register_input(cls)
                            importlib.import_module(fullname)
                except Exception as e:
                    # Ignore algorithms that could not be initialized.
                    logger.debug('Failed to load {} error {}'.format(fullname, str(e)))

        for item  in os.listdir(directory):
            fullpath = os.path.join(directory,item)
            if item not in _FOLDERS_TO_EXCLUDE and not item.endswith('dSYM') and os.path.isdir(fullpath):
                _discover_localinputs(fullpath,parentname + '.' + item)

    global _DISCOVERED
    _DISCOVERED = True
    syspath_save = sys.path
    sys.path = _get_sys_path(directory) + sys.path
    try:
        _discover_localinputs(directory,parentname)
    finally:
        sys.path = syspath_save

def register_input(cls, configuration=None):
    """
    Registers an input class
    Args:
        cls (object): Input class.
        configuration (object, optional): Pluggable configuration
    Returns:
        name: input name
    Raises:
        AlgorithmError: if the class is already registered or could not be registered
    """
    _discover_on_demand()

    # Verify that the pluggable is not already registered
    if cls in [input.cls for input in _REGISTERED_INPUTS.values()]:
        raise AlgorithmError('Could not register class {} is already registered'.format(cls))

    try:
        input_instance = cls(configuration=configuration)
    except Exception as err:
        raise AlgorithmError('Could not register input:{} could not be instantiated: {}'.format(cls, str(err)))

    # Verify that it has a minimal valid configuration.
    try:
        input_name = input_instance.configuration['name']
    except (LookupError, TypeError):
        raise AlgorithmError('Could not register input: invalid configuration')

    if input_name in _REGISTERED_INPUTS:
        raise AlgorithmError('Could not register class {}. Name {} {} is already registered'.format(cls,
                             input_name,_REGISTERED_INPUTS[input_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTERED_INPUTS[input_name] = RegisteredInput(input_name, cls, input_instance.configuration)
    return input_name

def deregister_input(input_name):
    """
    Deregisters am input class
    Args:
        input_name(str): The input name
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if input_name not in _REGISTERED_INPUTS:
        raise AlgorithmError('Could not deregister {} not registered'.format(input_name))

    _REGISTERED_INPUTS.pop(input_name)

def get_input_class(input_name):
    """
    Accesses input class
    Args:
        input_name (str): The input name
    Returns:
        cls: input class
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if input_name not in _REGISTERED_INPUTS:
        raise AlgorithmError('{} not registered'.format(input_name))

    return _REGISTERED_INPUTS[input_name].cls

def get_input_instance(input_name):
    """
    Instantiates an input class
    Args:
        input_name (str): The input name
    Returns:
        instance: input instance
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if input_name not in _REGISTERED_INPUTS:
        raise AlgorithmError('{} not registered'.format(input_name))

    return _REGISTERED_INPUTS[input_name].cls(configuration=_REGISTERED_INPUTS[input_name].configuration)

def get_input_configuration(input_name):
    """
    Accesses input configuration
    Args:
        input_name (str): The input name
    Returns:
        configuration: input configuration
    Raises:
        AlgorithmError: if the class is not registered
    """
    _discover_on_demand()

    if input_name not in _REGISTERED_INPUTS:
        raise AlgorithmError('{} not registered'.format(input_name))

    return _REGISTERED_INPUTS[input_name].configuration

def local_inputs():
    """
    Accesses input names
    Returns:
        names: input names
    """
    _discover_on_demand()
    return [input.name for input in _REGISTERED_INPUTS.values()]
