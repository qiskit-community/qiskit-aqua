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
from qiskit.backends.ibmq.credentials import (discover_credentials,
                                              store_credentials,
                                              Credentials)


class Preferences(object):

    _FILENAME = '.qiskit_aqua'
    _VERSION = '1.0'
    URL = 'https://quantumexperience.ng.bluemix.net/api'

    def __init__(self):
        """Create Preferences object."""
        self._preferences = {
            'version': Preferences._VERSION
        }
        self._packages_changed = False
        self._credentials_changed = False
        self._logging_config_changed = False
        self._token = None
        self._url = Preferences.URL
        self._proxy_urls = None

        credentials = discover_credentials()
        if credentials is not None:
            credentials = list(credentials.values())
            if len(credentials) > 0:
                credentials = credentials[0]
                self._token = credentials.token
                self._url = credentials.url
                if 'urls' in credentials.proxies:
                    self._proxy_urls = credentials.proxies['urls']

        home = os.path.expanduser("~")
        self._filepath = os.path.join(home, Preferences._FILENAME)
        try:
            with open(self._filepath) as json_pref:
                self._preferences = json.load(json_pref)
        except:
            pass

    def save(self):
        if self._credentials_changed:
            store_credentials(Credentials(
                self._token, self._url, proxies=self.get_proxies({})), overwrite=True)
            self._credentials_changed = False

        if self._logging_config_changed or self._packages_changed:
            with open(self._filepath, 'w') as fp:
                json.dump(self._preferences, fp, sort_keys=True, indent=4)
            self._logging_config_changed = False
            self._packages_changed = False

    def get_version(self):
        if 'version' in self._preferences:
            return self._preferences['version']

        return None

    def get_token(self, default_value=None):
        if self._token is not None:
            return self._token

        return default_value

    def set_token(self, token):
        if self._token != token:
            self._credentials_changed = True
            self._token = token

    def get_url(self, default_value=None):
        if self._url is not None:
            return self._url

        return default_value

    def set_url(self, url):
        if self._url != url:
            self._credentials_changed = True
            self._url = url

    def get_proxies(self, default_value=None):
        proxies = self.get_proxy_urls()
        if proxies is None:
            return default_value

        return {'urls': proxies}

    def get_proxy_urls(self, default_value=None):
        if self._proxy_urls is not None:
            return copy.deepcopy(self._proxy_urls)

        return default_value

    def set_proxy_urls(self, proxy_urls):
        if self._proxy_urls != proxy_urls:
            self._credentials_changed = True
            self._proxy_urls = proxy_urls

    def get_packages(self, default_value=None):
        if 'packages' in self._preferences and self._preferences['packages'] is not None:
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
