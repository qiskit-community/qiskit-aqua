# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import copy
from collections import OrderedDict
from qiskit.providers.ibmq.ibmqprovider import QE_URL
from qiskit.providers.ibmq.credentials import (read_credentials_from_qiskitrc,
                                               store_credentials,
                                               Credentials)
from qiskit.providers.ibmq.credentials.configrc import remove_credentials


class CredentialsPreferences(object):

    URL = QE_URL

    def __init__(self):
        """Create CredentialsPreferences object."""
        self._credentials_changed = False
        self._selected_credentials = None
        try:
            self._credentials = read_credentials_from_qiskitrc()
            if self._credentials is None:
                self._credentials = OrderedDict()
        except:
            self._credentials = OrderedDict()

        credentials = list(self._credentials.values())
        if len(credentials) > 0:
            self._selected_credentials = credentials[0]

    def save(self):
        if self._credentials_changed:
            try:
                dict = read_credentials_from_qiskitrc()
                if dict is not None:
                    for credentials in dict.values():
                        remove_credentials(credentials)
            except:
                self._credentials = OrderedDict()

            for credentials in self._credentials.values():
                store_credentials(credentials, overwrite=True)

            self._credentials_changed = False

    @property
    def credentials_changed(self):
        return self._credentials_changed

    @property
    def selected_credentials(self):
        return self._selected_credentials

    def get_all_credentials(self):
        return list(self._credentials.values())

    def get_credentials_with_same_key(self, url):
        if url is not None:
            credentials = Credentials('', url)
            return self._credentials.get(credentials.unique_id())
        return False

    def get_credentials(self, url):
        for credentials in self.get_all_credentials():
            if credentials.url == url:
                return credentials

        return None

    def set_credentials(self, token, url, proxy_urls=None):
        if url is not None and token is not None:
            proxies = {} if proxy_urls is None else {'urls': proxy_urls}
            credentials = Credentials(token, url, proxies=proxies)
            self._credentials[credentials.unique_id()] = credentials
            self._credentials_changed = True
            return credentials

        return None

    def select_credentials(self, url):
        if url is not None:
            credentials = Credentials('', url)
            if credentials.unique_id() in self._credentials:
                self._selected_credentials = self._credentials[credentials.unique_id()]

        return self._selected_credentials

    def remove_credentials(self, url):
        if url is not None:
            credentials = Credentials('', url)
            if credentials.unique_id() in self._credentials:
                del self._credentials[credentials.unique_id()]
                self._credentials_changed = True
            if self._selected_credentials is not None and self._selected_credentials.unique_id() == credentials.unique_id():
                self._selected_credentials = None
                credentials = list(self._credentials.values())
                if len(credentials) > 0:
                    self._selected_credentials = credentials[0]

    def get_token(self, default_value=None):
        if self._selected_credentials is not None:
            return self._selected_credentials.token

        return default_value

    def get_url(self, default_value=None):
        if self._selected_credentials is not None:
            return self._selected_credentials.url

        return default_value

    def get_proxies(self, default_value=None):
        if self._selected_credentials is not None:
            return copy.deepcopy(self._selected_credentials.proxies)

        return default_value

    def get_proxy_urls(self, default_value=None):
        if self._selected_credentials is not None and \
                'urls' in self._selected_credentials.proxies:
            return copy.deepcopy(self._selected_credentials.proxies['urls'])

        return default_value
