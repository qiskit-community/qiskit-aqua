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

"""Qiskit Aqua preferences"""

import os
import json
import copy
import logging

logger = logging.getLogger(__name__)


class Preferences:
    """Qiskit Aqua preferences"""

    _FILENAME = '.qiskit_aqua'
    _VERSION = '2.0'

    def __init__(self):
        """Create Preferences object."""
        self._preferences = {
            'version': Preferences._VERSION
        }
        self._ibmq_credentials_preferences = None

        home = os.path.expanduser("~")
        self._filepath = os.path.join(home, Preferences._FILENAME)
        try:
            with open(self._filepath) as json_pref:
                self._preferences = json.load(json_pref)
                # remove old no more valid entries
                if 'packages' in self._preferences:
                    del self._preferences['packages']
                if 'logging_config' in self._preferences:
                    del self._preferences['logging_config']
                if 'selected_ibmq_credentials_url' in self._preferences:
                    del self._preferences['selected_ibmq_credentials_url']
        except Exception:  # pylint: disable=broad-except
            pass

        self._old_preferences = copy.deepcopy(self._preferences)

    def save(self):
        """Saves Preferences"""
        if self._ibmq_credentials_preferences is not None:
            self._ibmq_credentials_preferences.save(self._preferences)
            if self._preferences != self._old_preferences:
                with open(self._filepath, 'w') as file:
                    json.dump(self._preferences, file, sort_keys=True, indent=4)

                self._old_preferences = copy.deepcopy(self._preferences)

    def get_version(self):
        """Return Preferences version"""
        if 'version' in self._preferences:
            return self._preferences['version']

        return None

    @property
    def ibmq_credentials_preferences(self):
        """Return IBMQ Credentials Preferences"""
        if self._ibmq_credentials_preferences is None:
            try:
                # pylint: disable=import-outside-toplevel
                from ._ibmq_credentials_preferences import IBMQCredentialsPreferences
                self._ibmq_credentials_preferences = IBMQCredentialsPreferences(self._preferences)
            except Exception as ex:  # pylint: disable=broad-except
                logger.debug("IBMQCredentialsPreferences not created: '%s'", str(ex))

        return self._ibmq_credentials_preferences
