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
import pkgutil
import importlib
import inspect
import copy
from ._basedriver import BaseDriver
from collections import namedtuple
from qiskit.chemistry import QiskitChemistryError
import pkg_resources

logger = logging.getLogger(__name__)

DRIVERS_ENTRY_POINT = 'qiskit.chemistry.drivers'

_NAMES_TO_EXCLUDE = [os.path.basename(__file__)]

_FOLDERS_TO_EXCLUDE = ['__pycache__']

RegisteredDriver = namedtuple(
    'RegisteredDriver', ['name', 'cls', 'configuration'])

_REGISTERED_DRIVERS = {}

_DISCOVERED = False


def refresh_drivers():
    """
    Attempts to rediscover all driver modules
    """
    global _REGISTERED_DRIVERS
    _REGISTERED_DRIVERS = {}
    global _DISCOVERED
    _DISCOVERED = True
    _discover_local_drivers()
    _discover_entry_point_chemistry_drivers()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Found: drivers {} ".format(local_drivers()))


def _discover_on_demand():
    """
    Attempts to discover drivers modules, if not already discovered
    """
    global _DISCOVERED
    if not _DISCOVERED:
        _DISCOVERED = True
        global _REGISTERED_DRIVERS
        _REGISTERED_DRIVERS = {}
        _discover_local_drivers()
        _discover_entry_point_chemistry_drivers()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Found: has drivers {} ".format(local_drivers()))


def _discover_entry_point_chemistry_drivers():
    """
    Discovers the chemistry driver modules defined by entry_points in setup
    and attempts to register them. Chem.Drivers modules should subclass BaseDriver Base class.
    """
    for entry_point in pkg_resources.iter_entry_points(DRIVERS_ENTRY_POINT):
        # first calls require and log any errors returned due to dependencies mismatches
        try:
            entry_point.require()
        except Exception as e:
            logger.warning("Entry point '{}' requirements issue: {}".format(entry_point, str(e)))

        # now  call resolve and try to load entry point
        try:
            ep = entry_point.resolve()
            _registered = False
            if not inspect.isabstract(ep) and issubclass(ep, BaseDriver):
                _register_driver(ep)
                _registered = True
                # print("Registered entry point chemistry driver '{}' class '{}'".format(entry_point, ep))
                logger.debug("Registered entry point chemistry driver '{}' class '{}'".format(entry_point, ep))
                break

            if not _registered:
                # print("Unknown entry point chemistry driver '{}' class '{}'".format(entry_point, ep))
                logger.debug("Unknown entry point chemistry driver '{}' class '{}'".format(entry_point, ep))
        except Exception as e:
            # Ignore entry point that could not be initialized.
            # print("Failed to load entry point '{}' error {}".format(entry_point, str(e)))
            logger.debug("Failed to load entry point '{}' error {}".format(entry_point, str(e)))


def _discover_local_drivers(directory=os.path.dirname(__file__),
                            parentname=os.path.splitext(__name__)[0],
                            names_to_exclude=_NAMES_TO_EXCLUDE,
                            folders_to_exclude=_FOLDERS_TO_EXCLUDE):
    """
    Discovers the chemistry drivers modules on the directory and subdirectories of the current module
    and attempts to register them. Driver modules should subclass BaseDriver Base class.
    Args:
        directory (str, optional): Directory to search for input modules. Defaults
            to the directory of this module.
        parentname (str, optional): Module parent name. Defaults to current directory name
        names_to_exclude (str, optional): File names to exclude. Defaults to _NAMES_TO_EXCLUDE
        folders_to_exclude (str, optional): Folders to exclude. Defaults to _FOLDERS_TO_EXCLUDE
    """
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
                           issubclass(cls, BaseDriver):
                            _register_driver(cls)
                            importlib.import_module(fullname)
                    except Exception as e:
                        # Ignore operator that could not be initialized.
                        logger.debug('Failed to load {} error {}'.format(fullname, str(e)))
            except Exception as e:
                # Ignore operator that could not be initialized.
                logger.debug('Failed to load {} error {}'.format(fullname, str(e)))

    for item in os.listdir(directory):
        fullpath = os.path.join(directory, item)
        if item not in folders_to_exclude and not item.endswith('dSYM') and os.path.isdir(fullpath):
            _discover_local_drivers(fullpath, parentname + '.' + item, names_to_exclude, folders_to_exclude)


def register_driver(cls):
    """
    Registers a driver class
    Args:
        cls (object): Driver class.
     Returns:
        name: driver name
    """
    _discover_on_demand()
    if not issubclass(cls, BaseDriver):
        raise QiskitChemistryError('Could not register class {} is not subclass of BaseDriver'.format(cls))

    return _register_driver(cls)


def _register_driver(cls):
    # Verify that the driver is not already registered.
    if cls in [driver.cls for driver in _REGISTERED_DRIVERS.values()]:
        raise QiskitChemistryError('Could not register class {} is already registered'.format(cls))

    # Verify that it has a minimal valid configuration.
    try:
        driver_name = cls.CONFIGURATION['name']
    except (LookupError, TypeError):
        raise QiskitChemistryError('Could not register driver: invalid configuration')

    # Verify that the driver is valid
    check_driver_valid = getattr(cls, 'check_driver_valid', None)
    if check_driver_valid is not None:
        try:
            check_driver_valid()
        except Exception as e:
            logger.debug(str(e))
            raise QiskitChemistryError('Could not register class {}. Name {} is not valid'.format(cls, driver_name)) from e

    if driver_name in _REGISTERED_DRIVERS:
        raise QiskitChemistryError('Could not register class {}. Name {} {} is already registered'.format(cls,
                                                                                                          driver_name,
                                                                                                          _REGISTERED_DRIVERS[driver_name].cls))

    # Append the driver to the `registered_classes` dict.
    _REGISTERED_DRIVERS[driver_name] = RegisteredDriver(driver_name, cls, copy.deepcopy(cls.CONFIGURATION))
    return driver_name


def deregister_driver(driver_name):
    """Remove driver from list of available drivers
    Args:
        driver_name (str): name of driver to unregister
    Raises:
        QiskitChemistryError if name is not registered.
    """
    _discover_on_demand()

    if driver_name not in _REGISTERED_DRIVERS:
        raise QiskitChemistryError('Could not deregister {} not registered'.format(driver_name))

    _REGISTERED_DRIVERS.pop(driver_name)


def get_driver_class(driver_name):
    """Return the class object for the named module.
    Args:
        driver_name (str): the module name
    Returns:
        Clas: class object for module
    Raises:
        QiskitChemistryError: if module is unavailable
    """
    _discover_on_demand()

    if driver_name not in _REGISTERED_DRIVERS:
        raise QiskitChemistryError('{} not registered'.format(driver_name))

    return _REGISTERED_DRIVERS[driver_name].cls


def get_driver_configuration(driver_name):
    """Return the configuration for the named module.
    Args:
        driver_name (str): the module name
    Returns:
        dict: configuration dict
    Raises:
        QiskitChemistryError: if module is unavailable
    """
    _discover_on_demand()

    if driver_name not in _REGISTERED_DRIVERS:
        raise QiskitChemistryError('{} not registered'.format(driver_name))

    return copy.deepcopy(_REGISTERED_DRIVERS[driver_name].configuration)


def local_drivers():
    """
    Accesses chemistry drivers names
    Returns:
        names: chemistry drivers names
    """
    _discover_on_demand()
    return [input.name for input in _REGISTERED_DRIVERS.values()]
