# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Driver Classes Discovery """

import os
import logging
import pkgutil
import importlib
import inspect
import copy
from collections import namedtuple
import pkg_resources
from qiskit.chemistry import QiskitChemistryError
from ._basedriver import BaseDriver

logger = logging.getLogger(__name__)

DRIVERS_ENTRY_POINT = 'qiskit.chemistry.drivers'

_NAMES_TO_EXCLUDE = [os.path.basename(__file__)]

_FOLDERS_TO_EXCLUDE = ['__pycache__', 'gauopen']


class RegistryChemDriver:
    """Contains Registered Chemistry Driver."""

    REGISTERED_DRIVER = namedtuple(
        'REGISTERED_DRIVER', ['name', 'cls', 'configuration'])

    def __init__(self) -> None:
        self._discovered = False
        self._registry = {}

    @property
    def discovered(self):
        """ Returns discovered flag """
        return self._discovered

    def set_discovered(self):
        """ Set registry as discovered """
        self._discovered = True

    @property
    def registry(self):
        """ Return registry dictionary """
        return self._registry

    def reset(self):
        """ reset registry data """
        self._discovered = False
        self._registry = {}


# Registry Global Instance
_REGISTRY_CHEM_DRIVER = RegistryChemDriver()


def refresh_drivers():
    """
    Attempts to rediscover all driver modules
    """
    _REGISTRY_CHEM_DRIVER.reset()
    _REGISTRY_CHEM_DRIVER.set_discovered()
    _discover_local_drivers()
    _discover_entry_pt_chem_drivers()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Found: drivers %s", local_drivers())


def _discover_on_demand():
    """
    Attempts to discover drivers modules, if not already discovered
    """
    if not _REGISTRY_CHEM_DRIVER.discovered:
        _REGISTRY_CHEM_DRIVER.reset()
        _REGISTRY_CHEM_DRIVER.set_discovered()
        _discover_local_drivers()
        _discover_entry_pt_chem_drivers()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Found: has drivers %s ", local_drivers())


def _discover_entry_pt_chem_drivers():
    """
    Discovers the chemistry driver modules defined by entry_points in setup
    and attempts to register them. Chem.Drivers modules should subclass BaseDriver Base class.
    """
    for entry_point in pkg_resources.iter_entry_points(DRIVERS_ENTRY_POINT):
        # first calls require and log any errors returned due to dependencies mismatches
        try:
            entry_point.require()
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning("Entry point '%s' requirements issue: %s", entry_point, str(ex))

        # now  call resolve and try to load entry point
        try:
            e_p = entry_point.resolve()
            _registered = False
            if not inspect.isabstract(e_p) and issubclass(e_p, BaseDriver):
                _register_driver(e_p)
                _registered = True
                logger.debug(
                    "Registered entry point chemistry driver '%s' class '%s'", entry_point, e_p)
                break

            if not _registered:
                logger.debug(
                    "Unknown entry point chemistry driver '%s' class '%s'", entry_point, e_p)
        except Exception as ex:  # pylint: disable=broad-except
            # Ignore entry point that could not be initialized.
            # print("Failed to load entry point '{}' error {}".format(entry_point, str(e)))
            logger.debug("Failed to load entry point '%s' error %s", entry_point, str(ex))


def _discover_local_drivers(directory=os.path.dirname(__file__),
                            parentname=os.path.splitext(__name__)[0],
                            names_to_exclude=None,
                            folders_to_exclude=None):
    """
    Discovers the chemistry drivers modules on the directory and
    subdirectories of the current module and attempts to register them.
    Driver modules should subclass BaseDriver Base class.
    Args:
        directory (Optional(str)): Directory to search for input modules. Defaults
            to the directory of this module.
        parentname (Optional(str)): Module parent name. Defaults to current directory name
        names_to_exclude (Optional(list[str])): File names to exclude.
                                                    Defaults to _NAMES_TO_EXCLUDE
        folders_to_exclude (Optional(list[str])): Folders to exclude.
                                                    Defaults to _FOLDERS_TO_EXCLUDE
    """
    names_to_exclude = names_to_exclude if names_to_exclude is not None else _NAMES_TO_EXCLUDE
    folders_to_exclude = folders_to_exclude \
        if folders_to_exclude is not None else _FOLDERS_TO_EXCLUDE
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
                    except Exception as ex:  # pylint: disable=broad-except
                        # Ignore operator that could not be initialized.
                        logger.debug('Failed to load %s error %s', fullname, str(ex))
            except Exception as ex:  # pylint: disable=broad-except
                # Ignore operator that could not be initialized.
                logger.debug('Failed to load %s error %s', fullname, str(ex))

    for item in os.listdir(directory):
        fullpath = os.path.join(directory, item)
        if item not in folders_to_exclude and \
                not item.endswith('dSYM') and os.path.isdir(fullpath):
            _discover_local_drivers(fullpath, parentname + '.' + item, names_to_exclude,
                                    folders_to_exclude)


def register_driver(cls):
    """
    Registers a driver class
    Args:
        cls (BaseDriver): Driver class.
     Returns:
        str: driver name
     Raises:
         QiskitChemistryError: if not derived from BaseDriver
    """
    _discover_on_demand()
    if not issubclass(cls, BaseDriver):
        raise QiskitChemistryError(
            'Could not register class {} is not subclass of BaseDriver'.format(cls))

    return _register_driver(cls)


def _register_driver(cls):
    # Verify that the driver is not already registered.
    if cls in [driver.cls for driver in _REGISTRY_CHEM_DRIVER.registry.values()]:
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
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug(str(ex))
            raise QiskitChemistryError(
                'Could not register class {}. Name {} is not valid'.format(
                    cls, driver_name)) from ex

    if driver_name in _REGISTRY_CHEM_DRIVER.registry:
        raise QiskitChemistryError(
            'Could not register class {}. Name {} {} is already registered'.format(
                cls, driver_name, _REGISTRY_CHEM_DRIVER.registry[driver_name].cls))

    # Append the driver to the `registered_classes` dict.
    _REGISTRY_CHEM_DRIVER.registry[driver_name] = \
        RegistryChemDriver.REGISTERED_DRIVER(driver_name, cls, copy.deepcopy(cls.CONFIGURATION))
    return driver_name


def deregister_driver(driver_name):
    """Remove driver from list of available drivers
    Args:
        driver_name (str): name of driver to unregister
    Raises:
        QiskitChemistryError: if name is not registered.
    """
    _discover_on_demand()

    if driver_name not in _REGISTRY_CHEM_DRIVER.registry:
        raise QiskitChemistryError('Could not deregister {} not registered'.format(driver_name))

    _REGISTRY_CHEM_DRIVER.registry.pop(driver_name)


def get_driver_class(driver_name):
    """Return the class object for the named module.
    Args:
        driver_name (str): the module name
    Returns:
        BaseDriver: class object for module
    Raises:
        QiskitChemistryError: if module is unavailable
    """
    _discover_on_demand()

    if driver_name not in _REGISTRY_CHEM_DRIVER.registry:
        raise QiskitChemistryError('{} not registered'.format(driver_name))

    return _REGISTRY_CHEM_DRIVER.registry[driver_name].cls


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

    if driver_name not in _REGISTRY_CHEM_DRIVER.registry:
        raise QiskitChemistryError('{} not registered'.format(driver_name))

    return copy.deepcopy(_REGISTRY_CHEM_DRIVER.registry[driver_name].configuration)


def local_drivers():
    """
    Accesses chemistry drivers names
    Returns:
        list[str]: chemistry drivers names
    """
    _discover_on_demand()
    return [input.name for input in _REGISTRY_CHEM_DRIVER.registry.values()]
