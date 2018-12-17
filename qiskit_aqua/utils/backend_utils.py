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

from qiskit import BasicAer, LegacySimulators
import logging
import warnings
import sys

logger = logging.getLogger(__name__)


def my_warning_wrapper(message, category, filename, lineno, file=None, line=None):
    msg = warnings.formatwarning(message, category, filename, lineno, line)
    # defaults deprecation warnings to logging
    if category == DeprecationWarning:
        logger.debug(msg)
    else:
        file = sys.stderr if file is None else file
        file.write(msg)


warnings.showwarning = my_warning_wrapper


def get_aer_backends():
    try:
        backends = LegacySimulators.backends()
        logger.debug('Using LegacySimulators backends.')
        return backends
    except:
        pass

    backends = BasicAer.backends()
    logger.debug('Using BasicAer backends.')
    return backends


def get_aer_backend(backend_name):
    try:
        backend = LegacySimulators.get_backend(backend_name)
        logger.debug('Using LegacySimulators backend {}.'.format(backend_name))
        return backend
    except:
        pass

    backend = BasicAer.get_backend(backend_name)
    logger.debug('Using BasicAer backend {}.'.format(backend_name))
    return backend
