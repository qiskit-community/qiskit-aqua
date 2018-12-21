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
from collections import OrderedDict
import sys
import pkgutil
import importlib
import inspect
import copy
from ._basedriver import BaseDriver
from qiskit_chemistry.preferences import Preferences
from collections import namedtuple
from qiskit_chemistry import QiskitChemistryError
import pkg_resources

logger = logging.getLogger(__name__)

_NAMES_TO_EXCLUDE = ['configurationmanager']

_FOLDERS_TO_EXCLUDE = ['__pycache__']

RegisteredDriver = namedtuple(
    'RegisteredDriver', ['name', 'cls', 'configuration'])

"""Singleton configuration class."""


class ConfigurationManager(object):

    __INSTANCE = None  # Shared instance

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

        def __init__(self):
            self._discovered = False
            self._registration = OrderedDict()

        def register_driver(self, cls):
            """
            Registers a driver class
            Args:
                cls (object): Driver class.
             Returns:
                name: driver name
            """
            self._discover_on_demand()
            if not issubclass(cls, BaseDriver):
                raise QiskitChemistryError(
                    'Could not register class {} is not subclass of BaseDriver'.format(cls))

            return self._register_driver(cls)

        def _register_driver(self, cls):
            # Verify that the driver is not already registered.
            if cls in [driver.cls for driver in self._registration.values()]:
                raise QiskitChemistryError(
                    'Could not register class {} is already registered'.format(cls))

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

            if driver_name in self._registration:
                raise QiskitChemistryError('Could not register class {}. Name {} {} is already registered'.format(cls,
                                                                                                                  driver_name,
                                                                                                                  self._registration[driver_name].cls))

            # Append the driver to the `registered_classes` dict.
            self._registration[driver_name] = RegisteredDriver(
                driver_name, cls, copy.deepcopy(cls.CONFIGURATION))
            return driver_name

        def deregister_driver(self, driver_name):
            """Remove driver from list of available drivers
            Args:
                driver_name (str): name of driver to unregister
            Raises:
                QiskitChemistryError if name is not registered.
            """
            self._discover_on_demand()

            if driver_name not in self._registration:
                raise QiskitChemistryError('Could not deregister {} not registered'.format(driver_name))

            self._registration.pop(driver_name)

        def get_driver_class(self, driver_name):
            """Return the class object for the named module.
            Args:
                driver_name (str): the module name
            Returns:
                Clas: class object for module
            Raises:
                QiskitChemistryError: if module is unavailable
            """
            self._discover_on_demand()

            if driver_name not in self._registration:
                raise QiskitChemistryError('{} not registered'.format(driver_name))

            return self._registration[driver_name].cls

        def get_driver_configuration(self, driver_name):
            """Return the configuration for the named module.
            Args:
                driver_name (str): the module name
            Returns:
                dict: configuration dict
            Raises:
                QiskitChemistryError: if module is unavailable
            """
            self._discover_on_demand()

            if driver_name not in self._registration:
                raise QiskitChemistryError('{} not registered'.format(driver_name))

            return copy.deepcopy(self._registration[driver_name].configuration)

        def get_driver_instance(self, name):
            """Return an instance for the name in configuration.
            Args:
                name (str): the name
            Returns:
                Object: module instance
            Raises:
                QiskitChemistryError: if module is unavailable
            """
            cls = self.get_driver_class(name)
            try:
                return cls()
            except Exception as err:
                raise QiskitChemistryError('{} could not be instantiated: {}'.format(cls, err))

        def local_drivers(self):
            """
            Accesses chemistry drivers names
            Returns:
                names: chemistry drivers names
            """
            self._discover_on_demand()
            return [input.name for input in self._registration.values()]

        def refresh_drivers(self):
            """
            Attempts to rediscover all driver modules
            """
            self._discovered = False
            self._registration = OrderedDict()
            self._discover_local_drivers()
            self._discover_entry_point_chemistry_drivers()
            self._discover_preferences_drivers()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Found: drivers {} ".format(self.local_drivers()))

        def _discover_on_demand(self):
            """
            Attempts to discover drivers modules, if not already discovered
            """
            if not self._discovered:
                self._discovered = True
                self._registration = OrderedDict()
                self._discover_local_drivers()
                self._discover_entry_point_chemistry_drivers()
                self._discover_preferences_drivers()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Found: has drivers {} ".format(self.local_drivers()))

        def _discover_entry_point_chemistry_drivers(self):
            """
            Discovers the chemistry driver modules defined by entry_points in setup
            and attempts to register them. Chem.Drivers modules should subclass BaseDriver Base class.
            """
            for entry_point in pkg_resources.iter_entry_points('qiskit.chemistry.drivers'):
                try:
                    ep = entry_point.load()
                    _registered = False
                    if issubclass(ep, BaseDriver):
                        self._register_driver(ep)
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

        def _discover_preferences_drivers(self):
            """
            Discovers the chemistry drivers on the directory and subdirectories of the preferences package
            and attempts to register them. Drivers modules should subclass BaseDriver Base class.
            """
            preferences = Preferences()
            packages = preferences.get_packages(Preferences.PACKAGE_TYPE_DRIVERS, [])
            for package in packages:
                try:
                    mod = importlib.import_module(package)
                    if mod is not None:
                        self._discover_local_drivers_in_dirs(os.path.dirname(mod.__file__),
                                                             mod.__name__,
                                                             names_to_exclude=[
                            '__main__'],
                            folders_to_exclude=['__pycache__'])
                    else:
                        # Ignore package that could not be initialized.
                        logger.debug('Failed to import package {}'.format(package))
                except Exception as e:
                    # Ignore package that could not be initialized.
                    logger.debug(
                        'Failed to load package {} error {}'.format(package, str(e)))

        def _discover_local_drivers_in_dirs(self,
                                            directory,
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
                                if cls.__module__ == modspec.name and issubclass(cls, BaseDriver):
                                    self._register_driver(cls)
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
                    self._discover_local_drivers_in_dirs(
                        fullpath, parentname + '.' + item, names_to_exclude, folders_to_exclude)

        def _discover_local_drivers(self,
                                   directory=os.path.dirname(__file__),
                                   parentname=os.path.splitext(__name__)[0]):
            """
            Discovers the chemistry drivers modules on the directory and subdirectories of the current module
            and attempts to register them. Driver modules should subclass BaseDriver Base class.
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
                self._discover_local_drivers_in_dirs(directory, parentname)
            finally:
                sys.path = syspath_save
