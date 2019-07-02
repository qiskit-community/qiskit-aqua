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


"""IBMQ Credential preferences"""

import copy
# pylint: disable=no-name-in-module, import-error
from qiskit.providers.ibmq.ibmqfactory import QX_AUTH_URL
from qiskit.providers.ibmq.credentials import Credentials
from qiskit.providers.ibmq.credentials.configrc import (read_credentials_from_qiskitrc,
                                                        store_credentials,
                                                        remove_credentials)


class IBMQCredentialsPreferences(object):

    IBMQ_URL = QX_AUTH_URL
    _IBMQ_KEY = 'ibmq'
    _HUB_KEY = 'hub'
    _GROUP_KEY = 'group'
    _PROJECT_KEY = 'project'

    def __init__(self, preferences_dict):
        """Create IBMQCredentialsPreferences object."""
        self._credentials_changed = False
        self._read_credentials()
        self._ibmq_dict = preferences_dict.get(IBMQCredentialsPreferences._IBMQ_KEY, {})
        self._ibmq_changed = False

    def _read_credentials(self):
        try:
            credentials = read_credentials_from_qiskitrc()
            if credentials:
                self._credentials = list(credentials.values())[0]
        except Exception:
            self._credentials = None

        self._old_credentials = self._credentials

    def save(self, preferences_dict):
        if self._credentials_changed:
            try:
                if self._credentials is not None:
                    store_credentials(self._credentials, overwrite=True)
                elif self._old_credentials is not None:
                    credentials_dict = read_credentials_from_qiskitrc()
                    if credentials_dict and self._old_credentials.unique_id() in credentials_dict:
                        remove_credentials(self._old_credentials)
            except Exception:
                pass

            self._credentials_changed = False
            self._read_credentials()

        if self._ibmq_changed:
            preferences_dict[IBMQCredentialsPreferences._IBMQ_KEY] = self._ibmq_dict
            self._ibmq_changed = False

    @property
    def credentials(self):
        return self._credentials

    def set_credentials(self, token, url=IBMQ_URL, proxy_urls=None):
        if url is not None and token is not None:
            proxies = {} if proxy_urls is None else {'urls': proxy_urls}
            self._credentials = Credentials(token, url, proxies=proxies)
            self._credentials_changed = True
            return self._credentials

        return None

    @property
    def url(self):
        if self._credentials is not None:
            return self._credentials.url

        return None

    @property
    def token(self):
        if self._credentials is not None:
            return self._credentials.token

        return None

    @property
    def proxies(self):
        if self._credentials is not None:
            return copy.deepcopy(self._credentials.proxies)

        return None

    @property
    def proxy_urls(self):
        if self._credentials is not None and \
                'urls' in self._credentials.proxies:
            return copy.deepcopy(self._credentials.proxies['urls'])

        return None

    @property
    def hub(self):
        return self._ibmq_dict.get(IBMQCredentialsPreferences._HUB_KEY)

    @hub.setter
    def hub(self, hub):
        """Set hub."""
        self._ibmq_dict[IBMQCredentialsPreferences._HUB_KEY] = hub
        self._ibmq_changed = True

    @property
    def group(self):
        return self._ibmq_dict.get(IBMQCredentialsPreferences._GROUP_KEY)

    @group.setter
    def group(self, group):
        """Set group."""
        self._ibmq_dict[IBMQCredentialsPreferences._GROUP_KEY] = group
        self._ibmq_changed = True

    @property
    def project(self):
        return self._ibmq_dict.get(IBMQCredentialsPreferences._PROJECT_KEY)

    @project.setter
    def project(self, project):
        """Set project."""
        self._ibmq_dict[IBMQCredentialsPreferences._PROJECT_KEY] = project
        self._ibmq_changed = True
