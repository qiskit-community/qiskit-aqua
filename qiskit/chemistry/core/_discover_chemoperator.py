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
from qiskit.chemistry import QiskitChemistryError
import logging
import sys
import copy
import pkg_resources

logger = logging.getLogger(__name__)

OPERATORS_ENTRY_POINT = 'qiskit.chemistry.operators'

_NAMES_TO_EXCLUDE = [os.path.basename(__file__)]

_FOLDERS_TO_EXCLUDE = ['__pycache__']

RegisteredChemOp = namedtuple(
    'RegisteredChemOp', ['name', 'cls', 'configuration'])

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
    _discover_local_chemistry_operators()
    _discover_entry_point_chemistry_operators()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Found: chemistry operators {} ".format(
            local_chemistry_operators()))


def _discover_on_demand():
    """
    Attempts to discover operator modules, if not already discovered
    """
    global _DISCOVERED
    if not _DISCOVERED:
        _DISCOVERED = True
        global _REGISTERED_CHEMISTRY_OPERATORS
        _REGISTERED_CHEMISTRY_OPERATORS = {}
        _discover_local_chemistry_operators()
        _discover_entry_point_chemistry_operators()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Found: chemistry operators {} ".format(
                local_chemistry_operators()))


def _discover_entry_point_chemistry_operators():
    """
    Discovers the chemistry operators modules defined by entry_points in setup
    and attempts to register them. Chem.Operator modules should subclass ChemistryOperator Base class.
    """
    for entry_point in pkg_resources.iter_entry_points(OPERATORS_ENTRY_POINT):
        # first calls require and log any errors returned due to dependencies mismatches
        try:
            entry_point.require()
        except Exception as e:
            logger.warning("Entry point '{}' requirements issue: {}".format(entry_point, str(e)))

        # now  call resolve and try to load entry point
        try:
            ep = entry_point.resolve()
            _registered = False
            if not inspect.isabstract(ep) and issubclass(ep, ChemistryOperator):
                register_chemistry_operator(ep)
                _registered = True
                # print("Registered entry point chemistry operator '{}' class '{}'".format(entry_point, ep))
                logger.debug("Registered entry point chemistry operator '{}' class '{}'".format(entry_point, ep))
                break

            if not _registered:
                # print("Unknown entry point chemistry operator '{}' class '{}'".format(entry_point, ep))
                logger.debug("Unknown entry point chemistry operator '{}' class '{}'".format(entry_point, ep))
        except Exception as e:
            # Ignore entry point that could not be initialized.
            # print("Failed to load entry point '{}' error {}".format(entry_point, str(e)))
            logger.debug("Failed to load entry point '{}' error {}".format(entry_point, str(e)))


def _discover_local_chemistry_operators_in_dirs(directory,
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
                        if cls.__module__ == modspec.name and \
                           not inspect.isabstract(cls) and \
                           issubclass(cls, ChemistryOperator):
                            _register_chemistry_operator(cls)
                            importlib.import_module(fullname)
                    except Exception as e:
                        # Ignore operator that could not be initialized.
                        logger.debug(
                            'Failed to load {} error {}'.format(fullname, str(e)))
            except Exception as e:
                # Ignore operator that could not be initialized.
                logger.debug(
                    'Failed to load {} error {}'.format(fullname, str(e)))

    for item in os.listdir(directory):
        fullpath = os.path.join(directory, item)
        if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
            _discover_local_chemistry_operators_in_dirs(
                fullpath, parentname + '.' + item, names_to_exclude, folders_to_exclude)


def _discover_local_chemistry_operators(directory=os.path.dirname(__file__),
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
        for item in os.listdir(directory):
            fullpath = os.path.join(directory, item)
            if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                syspath += _get_sys_path(fullpath)

        return syspath

    syspath_save = sys.path
    sys.path = _get_sys_path(directory) + sys.path
    try:
        _discover_local_chemistry_operators_in_dirs(directory, parentname)
    finally:
        sys.path = syspath_save


def register_chemistry_operator(cls):
    """
    Registers a chemistry operator class
    Args:
        cls (object): chemistry operator class.
    Returns:
        name: input name
    Raises:
        QiskitChemistryError: if the class is already registered or could not be registered
    """
    _discover_on_demand()
    if not issubclass(cls, ChemistryOperator):
        raise QiskitChemistryError('Could not register class {} is not subclass of ChemistryOperator'.format(cls))

    return _register_chemistry_operator(cls)


def _register_chemistry_operator(cls):
    # Verify that the pluggable is not already registered
    if cls in [input.cls for input in _REGISTERED_CHEMISTRY_OPERATORS.values()]:
        raise QiskitChemistryError('Could not register class {} is already registered'.format(cls))

    # Verify that it has a minimal valid configuration.
    try:
        chemistry_operator_name = cls.CONFIGURATION['name']
    except (LookupError, TypeError):
        raise QiskitChemistryError('Could not register chemistry operator: invalid configuration')

    if chemistry_operator_name in _REGISTERED_CHEMISTRY_OPERATORS:
        raise QiskitChemistryError('Could not register class {}. Name {} {} is already registered'.format(cls,
                                                                                                          chemistry_operator_name, _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].cls))

    # Append the pluggable to the `registered_classes` dict.
    _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name] = RegisteredChemOp(
        chemistry_operator_name, cls, copy.deepcopy(cls.CONFIGURATION))
    return chemistry_operator_name


def deregister_chemistry_operator(chemistry_operator_name):
    """
    Deregisters a chemistry operator class
    Args:
        chemistry_operator_name(str): The chemistry operator name
    Raises:
        QiskitChemistryError: if the class is not registered
    """
    _discover_on_demand()

    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise QiskitChemistryError(
            'Could not deregister {} not registered'.format(chemistry_operator_name))

    _REGISTERED_CHEMISTRY_OPERATORS.pop(chemistry_operator_name)


def get_chemistry_operator_class(chemistry_operator_name):
    """
    Accesses chemistry operator class
    Args:
        chemistry_operator_name (str): The chemistry operator name
    Returns:
        cls: chemistry operator class
    Raises:
        QiskitChemistryError: if the class is not registered
    """
    _discover_on_demand()

    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise QiskitChemistryError(
            '{} not registered'.format(chemistry_operator_name))

    return _REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].cls


def get_chemistry_operator_configuration(chemistry_operator_name):
    """
    Accesses chemistry operator configuration
    Args:
        chemistry_operator_name (str): The chemistry operator name
    Returns:
        configuration: chemistry operator configuration
    Raises:
        QiskitChemistryError: if the class is not registered
    """
    _discover_on_demand()

    if chemistry_operator_name not in _REGISTERED_CHEMISTRY_OPERATORS:
        raise QiskitChemistryError('{} not registered'.format(chemistry_operator_name))

    return copy.deepcopy(_REGISTERED_CHEMISTRY_OPERATORS[chemistry_operator_name].configuration)


def local_chemistry_operators():
    """
    Accesses chemistry operator names
    Returns:
        names: chemistry operator names
    """
    _discover_on_demand()
    return [input.name for input in _REGISTERED_CHEMISTRY_OPERATORS.values()]
