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
import logging
from collections import OrderedDict
# pylint: disable=no-name-in-module, import-error
from qiskit.providers.ibmq.ibmqfactory import QX_AUTH_URL
from qiskit.providers.ibmq.credentials import Credentials
# pylint: disable=syntax-error
from qiskit.providers.ibmq.credentials.configrc import (read_credentials_from_qiskitrc,
                                                        write_qiskit_rc)
from qiskit.providers.ibmq.credentials.updater import is_directly_updatable, QE2_AUTH_URL

logger = logging.getLogger(__name__)


class IBMQCredentialsPreferences:
    """IBMQ Credential preferences"""
    _IBMQ_KEY = 'ibmq'
    _HUB_KEY = 'hub'
    _GROUP_KEY = 'group'
    _PROJECT_KEY = 'project'

    def __init__(self, preferences_dict):
        """Create IBMQCredentialsPreferences object."""
        self._ibmq_dict = preferences_dict.get(IBMQCredentialsPreferences._IBMQ_KEY, {})
        self._ibmq_changed = False
        self._credentials_changed = False
        self._credentials = None
        self._read_credentials()

    def _read_credentials(self):
        """Read first credential from file and attempt to convert to v2"""
        self._credentials = None
        try:
            credentials = read_credentials_from_qiskitrc()
            if credentials:
                credentials = list(credentials.values())[0]
                if credentials:
                    if is_directly_updatable(credentials):
                        self._credentials = Credentials(credentials.token,
                                                        QE2_AUTH_URL,
                                                        proxies=credentials.proxies,
                                                        verify=credentials.verify)
                    elif credentials.url == QE2_AUTH_URL:
                        self._credentials = credentials
                    elif credentials.is_ibmq():
                        self._credentials = Credentials(credentials.token,
                                                        QE2_AUTH_URL,
                                                        proxies=credentials.proxies,
                                                        verify=credentials.verify)
                        self._ibmq_dict[IBMQCredentialsPreferences._HUB_KEY] = credentials.hub
                        self._ibmq_dict[IBMQCredentialsPreferences._GROUP_KEY] = credentials.group
                        self._ibmq_dict[IBMQCredentialsPreferences._PROJECT_KEY] = \
                            credentials.project
                    else:
                        # Unknown URL - do not act on it.
                        logger.debug('The stored account with url "%s" could not be '
                                     'parsed.', credentials.url)
        except Exception as ex:  # pylint: disable=broad-except
            logger.debug("Failed to read IBM credentials: '%s'", str(ex))

    def save(self, preferences_dict):
        """Save credentials, always keep only one"""
        if self._credentials_changed:
            try:
                stored_credentials = OrderedDict()
                if self._credentials is not None:
                    stored_credentials[self._credentials.unique_id()] = self._credentials

                write_qiskit_rc(stored_credentials)
            except Exception as ex:  # pylint: disable=broad-except
                logger.debug("Failed to store IBM credentials: '%s'", str(ex))

            self._credentials_changed = False
            self._read_credentials()

        if self._ibmq_changed:
            preferences_dict[IBMQCredentialsPreferences._IBMQ_KEY] = self._ibmq_dict
            self._ibmq_changed = False

    @property
    def credentials(self):
        """ returns credentials """
        return self._credentials

    def set_credentials(self, token, proxy_urls=None):
        """ set credentials """
        if token is not None:
            proxies = {} if proxy_urls is None else {'urls': proxy_urls}
            cred = Credentials(token, QX_AUTH_URL, proxies=proxies)
            if self._credentials is None or self._credentials != cred:
                self._credentials = cred
                self._credentials_changed = True
        else:
            if self._credentials is not None:
                self._credentials_changed = True

            self._credentials = None
            self.hub = None
            self.group = None
            self.project = None

        return self._credentials

    @property
    def url(self):
        """ returns URL """
        if self._credentials is not None:
            return self._credentials.url

        return None

    @property
    def token(self):
        """ returns token """
        if self._credentials is not None:
            return self._credentials.token

        return None

    @property
    def proxies(self):
        """ returns proxies """
        if self._credentials is not None:
            return copy.deepcopy(self._credentials.proxies)

        return None

    @property
    def proxy_urls(self):
        """ returns proxy URL list """
        if self._credentials is not None and \
                'urls' in self._credentials.proxies:
            return copy.deepcopy(self._credentials.proxies['urls'])

        return None

    @property
    def hub(self):
        """ return hub """
        return self._ibmq_dict.get(IBMQCredentialsPreferences._HUB_KEY)

    @hub.setter
    def hub(self, hub):
        """Set hub."""
        self._ibmq_dict[IBMQCredentialsPreferences._HUB_KEY] = hub
        self._ibmq_changed = True

    @property
    def group(self):
        """ returns group """
        return self._ibmq_dict.get(IBMQCredentialsPreferences._GROUP_KEY)

    @group.setter
    def group(self, group):
        """Set group."""
        self._ibmq_dict[IBMQCredentialsPreferences._GROUP_KEY] = group
        self._ibmq_changed = True

    @property
    def project(self):
        """ returns project """
        return self._ibmq_dict.get(IBMQCredentialsPreferences._PROJECT_KEY)

    @project.setter
    def project(self, project):
        """Set project."""
        self._ibmq_dict[IBMQCredentialsPreferences._PROJECT_KEY] = project
        self._ibmq_changed = True
