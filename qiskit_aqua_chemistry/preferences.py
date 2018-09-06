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
import copy
from qiskit_aqua_chemistry import AquaChemistryError


class Preferences(object):

    PACKAGE_TYPE_DRIVERS = 'drivers'
    PACKAGE_TYPE_CHEMISTRY = 'chemistry'
    _FILENAME = '.qiskit_aqua_chemistry'
    _VERSION = '1.0'

    def __init__(self):
        """Create Preferences object."""
        self._preferences = {
            'version': Preferences._VERSION
        }
        self._packages_changed = False
        self._logging_config_changed = False

        home = os.path.expanduser("~")
        self._filepath = os.path.join(home, Preferences._FILENAME)
        try:
            with open(self._filepath) as json_pref:
                self._preferences = json.load(json_pref)
        except:
            pass

    def save(self):
        if self._logging_config_changed or self._packages_changed:
            with open(self._filepath, 'w') as fp:
                json.dump(self._preferences, fp, sort_keys=True, indent=4)
            self._logging_config_changed = False
            self._packages_changed = False

    def get_version(self):
        if 'version' in self._preferences:
            return self._preferences['version']

        return None

    def get_packages(self, package_type, default_value=None):
        if package_type is not None and isinstance(package_type, str) and \
            'packages' in self._preferences and self._preferences['packages'] is not None and \
                package_type in self._preferences['packages'] and self._preferences['packages'][package_type] is not None:
            return copy.deepcopy(self._preferences['packages'][package_type])

        return default_value

    def add_package(self, package_type, package):
        if package_type is not None and isinstance(package_type, str) and package is not None and isinstance(package, str):
            if package_type != Preferences.PACKAGE_TYPE_DRIVERS and package_type != Preferences.PACKAGE_TYPE_CHEMISTRY:
                raise AquaChemistryError(
                    'Invalid package type {}'.format(package_type))

            packages = self.get_packages(package_type, [])
            if package not in packages:
                packages.append(package)
                if 'packages' in self._preferences and self._preferences['packages'] is not None:
                    self._preferences['packages'][package_type] = packages
                else:
                    self._preferences['packages'] = {package_type: packages}

                self._packages_changed = True
                return True

        return False

    def change_package(self, package_type, old_package, new_package):
        if package_type is not None and isinstance(package_type, str) and \
                old_package is not None and isinstance(old_package, str) and \
                new_package is not None and isinstance(new_package, str):
            if package_type != Preferences.PACKAGE_TYPE_DRIVERS and package_type != Preferences.PACKAGE_TYPE_CHEMISTRY:
                raise AquaChemistryError(
                    'Invalid package type {}'.format(package_type))

            packages = self.get_packages(package_type, [])
            for index, package in enumerate(packages):
                if package == old_package:
                    packages[index] = new_package
                    if 'packages' in self._preferences and self._preferences['packages'] is not None:
                        self._preferences['packages'][package_type] = packages
                    else:
                        self._preferences['packages'] = {
                            package_type: packages}

                    self._packages_changed = True
                    return True

        return False

    def remove_package(self, package_type, package):
        if package_type is not None and isinstance(package_type, str) and package is not None and isinstance(package, str):
            packages = self.get_packages(package_type, [])
            if package in packages:
                packages.remove(package)
                if 'packages' in self._preferences and self._preferences['packages'] is not None:
                    self._preferences['packages'][package_type] = packages
                else:
                    self._preferences['packages'] = {package_type: packages}

                self._packages_changed = True
                return True

        return False

    def set_packages(self, package_type, packages):
        if package_type is not None and isinstance(package_type, str):
            if package_type != Preferences.PACKAGE_TYPE_DRIVERS and package_type != Preferences.PACKAGE_TYPE_CHEMISTRY:
                raise AquaChemistryError(
                    'Invalid package type {}'.format(package_type))

            if 'packages' in self._preferences and self._preferences['packages'] is not None:
                self._preferences['packages'][package_type] = packages
            else:
                self._preferences['packages'] = {package_type: packages}

            self._packages_changed = True
            return True

        return False

        self._packages_changed = True
        self._preferences['packages'] = packages

    def get_logging_config(self, default_value=None):
        if 'logging_config' in self._preferences:
            return self._preferences['logging_config']

        return default_value

    def set_logging_config(self, logging_config):
        self._logging_config_changed = True
        self._preferences['logging_config'] = logging_config
