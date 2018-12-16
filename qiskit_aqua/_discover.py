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

import logging
import sys
import os
import pkgutil
import importlib
import inspect
import copy
from collections import namedtuple
from enum import Enum
from qiskit_aqua import AquaError

logger = logging.getLogger(__name__)


class PluggableType(Enum):
    ALGORITHM = 'algorithm'
    OPTIMIZER = 'optimizer'
    VARIATIONAL_FORM = 'variational_form'
    INITIAL_STATE = 'initial_state'
    IQFT = 'iqft'
    ORACLE = 'oracle'
    FEATURE_MAP = 'feature_map'
    MULTICLASS_EXTENSION = 'multiclass_extension'
    UNCERTAINTY_PROBLEM = 'uncertainty_problem'
    UNCERTAINTY_MODEL = 'uncertainty_model'
    INPUT = 'input'


def _get_pluggables_types_dictionary():
    """
    Gets all the pluggables types
    Any new pluggable type should be added here
    """
    from qiskit_aqua.components.uncertainty_problems import UncertaintyProblem
    from qiskit_aqua.components.random_distributions import RandomDistribution
    from qiskit_aqua.components.optimizers import Optimizer
    from qiskit_aqua.algorithms.quantum_algorithm import QuantumAlgorithm
    from qiskit_aqua.components.variational_forms import VariationalForm
    from qiskit_aqua.components.initial_states import InitialState
    from qiskit_aqua.components.iqfts import IQFT
    from qiskit_aqua.components.oracles import Oracle
    from qiskit_aqua.components.feature_maps import FeatureMap
    from qiskit_aqua.components.multiclass_extensions import MulticlassExtension
    from qiskit_aqua.input import AlgorithmInput
    return {
        PluggableType.ALGORITHM: QuantumAlgorithm,
        PluggableType.OPTIMIZER: Optimizer,
        PluggableType.VARIATIONAL_FORM: VariationalForm,
        PluggableType.INITIAL_STATE: InitialState,
        PluggableType.IQFT: IQFT,
        PluggableType.ORACLE: Oracle,
        PluggableType.FEATURE_MAP: FeatureMap,
        PluggableType.MULTICLASS_EXTENSION: MulticlassExtension,
        PluggableType.UNCERTAINTY_PROBLEM: UncertaintyProblem,
        PluggableType.UNCERTAINTY_MODEL: RandomDistribution,
        PluggableType.INPUT: AlgorithmInput
    }


_NAMES_TO_EXCLUDE = [
    '_aqua',
    '_discover',
    '_logging',
    'aqua_error',
    'operator',
    'pluggable',
    'quantum_instance',
    'optimizer',
    'variational_form',
    'initial_state',
    'iqft',
    'oracle',
    'feature_map',
    'multiclass_extension',
    'uncertainty_problem',
    'uncertainty_model',
    'univariate_uncertainty_model'
]

_FOLDERS_TO_EXCLUDE = [
    '__pycache__',
    'parser',
    'translators',
    'utils'
]

RegisteredPluggable = namedtuple(
    'RegisteredPluggable', ['name', 'cls', 'configuration'])

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
            logger.debug("Found: '{}' has pluggables {} ".format(ptype.value, local_pluggables(ptype)))


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
                logger.debug("Found: '{}' has pluggables {} ".format(ptype.value, local_pluggables(ptype)))


def discover_preferences_pluggables():
    """
    Discovers the pluggable modules on the directory and subdirectories of the preferences package
    and attempts to register them. Pluggable modules should subclass Pluggable Base classes.
    """
    from qiskit_aqua_cmd import Preferences
    preferences = Preferences()
    packages = preferences.get_packages([])
    for package in packages:
        try:
            mod = importlib.import_module(package)
            if mod is not None:
                _discover_local_pluggables(os.path.dirname(mod.__file__),
                                           mod.__name__,
                                           names_to_exclude=['__main__'],
                                           folders_to_exclude=['__pycache__'])
            else:
                # Ignore package that could not be initialized.
                # print('Failed to import package {}'.format(package))
                logger.debug('Failed to import package {}'.format(package))
        except Exception as e:
            # Ignore package that could not be initialized.
            # print('Failed to load package {} error {}'.format(package, str(e)))
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
                            for pluggable_type, c in _get_pluggables_types_dictionary().items():
                                if issubclass(cls, c):
                                    _register_pluggable(pluggable_type, cls)
                                    importlib.import_module(fullname)
                                    break
                    except Exception as e:
                        # Ignore pluggables that could not be initialized.
                        # print('Failed to load pluggable {} error {}'.format(fullname, str(e)))
                        logger.debug('Failed to load pluggable {} error {}'.format(fullname, str(e)))

            except Exception as e:
                # Ignore pluggables that could not be initialized.
                # print('Failed to load {} error {}'.format(fullname, str(e)))
                logger.debug('Failed to load {} error {}'.format(fullname, str(e)))

    for item in sorted(os.listdir(directory)):
        fullpath = os.path.join(directory, item)
        if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
            _discover_local_pluggables(
                fullpath, parentname + '.' + item, names_to_exclude, folders_to_exclude)


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
        for item in os.listdir(directory):
            fullpath = os.path.join(directory, item)
            if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                syspath += _get_sys_path(fullpath)

        return syspath

    syspath_save = sys.path
    sys.path = sys.path + _get_sys_path(directory)
    try:
        _discover_local_pluggables(directory, parentname)
    finally:
        sys.path = syspath_save


def register_pluggable(cls):
    """
    Registers a pluggable class
    Args:
        cls (object): Pluggable class.
     Returns:
        name: pluggable name
    """
    _discover_on_demand()
    pluggable_type = None
    for type, c in _get_pluggables_types_dictionary().items():
        if issubclass(cls, c):
            pluggable_type = type
            break

    if pluggable_type is None:
        raise AquaError(
            'Could not register class {} is not subclass of any known pluggable'.format(cls))

    return _register_pluggable(pluggable_type, cls)


global_class = None


def _register_pluggable(pluggable_type, cls):
    """
    Registers a pluggable class
    Args:
        pluggable_type(PluggableType): The pluggable type
        cls (object): Pluggable class.
     Returns:
        name: pluggable name
    Raises:
        AquaError: if the class is already registered or could not be registered
    """
    if pluggable_type not in _REGISTERED_PLUGGABLES:
        _REGISTERED_PLUGGABLES[pluggable_type] = {}

    # fix pickle problems
    method = 'from {} import {}\nglobal global_class\nglobal_class = {}'.format(cls.__module__, cls.__qualname__, cls.__qualname__)
    exec(method)
    cls = global_class

    # Verify that the pluggable is not already registered.
    registered_classes = _REGISTERED_PLUGGABLES[pluggable_type]
    if cls in [pluggable.cls for pluggable in registered_classes.values()]:
        raise AquaError(
            'Could not register class {} is already registered'.format(cls))

    # Verify that it has a minimal valid configuration.
    try:
        pluggable_name = cls.CONFIGURATION['name']
    except (LookupError, TypeError):
        raise AquaError('Could not register pluggable: invalid configuration')

    # Verify that the pluggable is valid
    check_pluggable_valid = getattr(cls, 'check_pluggable_valid', None)
    if check_pluggable_valid is not None:
        try:
            check_pluggable_valid()
        except Exception as e:
            logger.debug(str(e))
            raise AquaError('Could not register class {}. Name {} is not valid'.format(cls, pluggable_name)) from e

    if pluggable_name in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AquaError('Could not register class {}. Name {} {} is already registered'.format(cls,
                                                                                               pluggable_name, _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name] = RegisteredPluggable(
        pluggable_name, cls, copy.deepcopy(cls.CONFIGURATION))
    return pluggable_name


def deregister_pluggable(pluggable_type, pluggable_name):
    """
    Deregisters a pluggable class
    Args:
        pluggable_type(PluggableType): The pluggable type
        pluggable_name (str): The pluggable name
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AquaError('Could not deregister {} {} not registered'.format(
            pluggable_type, pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AquaError('Could not deregister {} {} not registered'.format(
            pluggable_type, pluggable_name))

    _REGISTERED_PLUGGABLES[pluggable_type].pop(pluggable_name)


def get_pluggable_class(pluggable_type, pluggable_name):
    """
    Accesses pluggable class
    Args:
        pluggable_type(PluggableType or str): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        cls: pluggable class
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError('Invalid pluggable type {} {}'.format(
            pluggable_type, pluggable_name))

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AquaError('{} {} not registered'.format(
            pluggable_type, pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AquaError('{} {} not registered'.format(
            pluggable_type, pluggable_name))

    return _REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].cls


def get_pluggable_configuration(pluggable_type, pluggable_name):
    """
    Accesses pluggable configuration
    Args:
        pluggable_type(PluggableType or str): The pluggable type
        pluggable_name (str): The pluggable name
    Returns:
        configuration: pluggable configuration
    Raises:
        AquaError: if the class is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError('Invalid pluggable type {} {}'.format(
            pluggable_type, pluggable_name))

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AquaError('{} {} not registered'.format(
            pluggable_type, pluggable_name))

    if pluggable_name not in _REGISTERED_PLUGGABLES[pluggable_type]:
        raise AquaError('{} {} not registered'.format(
            pluggable_type, pluggable_name))

    return copy.deepcopy(_REGISTERED_PLUGGABLES[pluggable_type][pluggable_name].configuration)


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
        pluggable_type(PluggableType or str): The pluggable type
    Returns:
        names: pluggable names
    Raises:
        AquaError: if the tyoe is not registered
    """
    _discover_on_demand()

    if isinstance(pluggable_type, str):
        for ptype in PluggableType:
            if ptype.value == pluggable_type:
                pluggable_type = ptype
                break

    if not isinstance(pluggable_type, PluggableType):
        raise AquaError(
            'Invalid pluggable type {}'.format(pluggable_type))

    if pluggable_type not in _REGISTERED_PLUGGABLES:
        raise AquaError('{} not registered'.format(pluggable_type))

    return [pluggable.name for pluggable in _REGISTERED_PLUGGABLES[pluggable_type].values()]
