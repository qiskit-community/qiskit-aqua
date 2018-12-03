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


class Preferences(object):

    _FILENAME = '.qiskit_aqua'
    _VERSION = '1.0'
    _SELECTED_KEY = 'selected_ibmq_credentials_url'

    def __init__(self):
        """Create Preferences object."""
        self._preferences = {
            'version': Preferences._VERSION
        }
        self._packages_changed = False
        self._logging_config_changed = False
        self._credentials_preferences = None

        home = os.path.expanduser("~")
        self._filepath = os.path.join(home, Preferences._FILENAME)
        try:
            with open(self._filepath) as json_pref:
                self._preferences = json.load(json_pref)
        except:
            pass

    def save(self):
        pref_changed = self._logging_config_changed or self._packages_changed
        if self._credentials_preferences is not None:
            self.credentials_preferences.save()
            selected_credentials = self.credentials_preferences.selected_credentials
            selected_credentials_url = selected_credentials.url if selected_credentials is not None else None

            if selected_credentials_url != self._preferences.get(Preferences._SELECTED_KEY) or pref_changed:
                pref_changed = True
                selected_credentials = self.credentials_preferences.selected_credentials
                if selected_credentials_url is not None:
                    self._preferences[Preferences._SELECTED_KEY] = selected_credentials_url
                else:
                    if Preferences._SELECTED_KEY in self._preferences:
                        del self._preferences[Preferences._SELECTED_KEY]

        if pref_changed:
            with open(self._filepath, 'w') as fp:
                json.dump(self._preferences, fp, sort_keys=True, indent=4)

        self._logging_config_changed = False
        self._packages_changed = False

    def get_version(self):
        if 'version' in self._preferences:
            return self._preferences['version']

        return None

    @property
    def credentials_preferences(self):
        """Return credentials preferences"""
        from ._credentialspreferences import CredentialsPreferences
        if self._credentials_preferences is None:
            self._credentials_preferences = CredentialsPreferences()
            if Preferences._SELECTED_KEY in self._preferences:
                self._credentials_preferences.select_credentials(self._preferences[Preferences._SELECTED_KEY])

        return self._credentials_preferences

    def get_token(self, default_value=None):
        return self.credentials_preferences.get_token(default_value)

    def get_url(self, default_value=None):
        return self.credentials_preferences.get_url(default_value)

    def get_proxies(self, default_value=None):
        return self.credentials_preferences.get_proxies(default_value)

    def get_proxy_urls(self, default_value=None):
        return self.credentials_preferences.get_proxy_urls(default_value)

    def get_packages(self, default_value=None):
        if 'packages' in self._preferences and \
           self._preferences['packages'] is not None:
            return copy.deepcopy(self._preferences['packages'])

        return default_value

    def add_package(self, package):
        if package is not None and isinstance(package, str):
            packages = self.get_packages([])
            if package not in packages:
                packages.append(package)
                self.set_packages(packages)

    def remove_package(self, package):
        if package is not None and isinstance(package, str):
            packages = self.get_packages([])
            if package in packages:
                packages.remove(package)
                self.set_packages(packages)

    def set_packages(self, packages):
        self._packages_changed = True
        self._preferences['packages'] = packages

    def get_logging_config(self, default_value=None):
        if 'logging_config' in self._preferences:
            return self._preferences['logging_config']

        return default_value

    def set_logging_config(self, logging_config):
        self._logging_config_changed = True
        self._preferences['logging_config'] = logging_config
