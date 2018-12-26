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

from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq.ibmqsingleprovider import IBMQSingleProvider
from qiskit.providers.ibmq.credentials import Credentials
from qiskit_aqua_cmd import Preferences
import sys
import warnings
import logging
from collections import OrderedDict
import importlib

logger = logging.getLogger(__name__)

_UNSUPPORTED_BACKENDS = ['unitary_simulator', 'clifford_simulator']


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
        from qiskit import Aer
        backends = Aer.backends()
    except:
        backends = BasicAer.backends()
    return backends


def get_aer_backend(backend_name):
    try:
        from qiskit import Aer
        backend = Aer.get_backend(backend_name)
    except:
        backend = BasicAer.get_backend(backend_name)
    return backend


def get_backend_from_provider(provider_name, backend_name):
    index = provider_name.rfind(".")
    if index < 1:
        raise ImportError("Invalid provider name '{}'".format(provider_name))

    modulename = provider_name[0:index]
    objectname = provider_name[index + 1:len(provider_name)]

    module = importlib.import_module(modulename)
    if module is None:
        raise ImportError("Failed to import provider '{}'".format(provider_name))

    provider_object = getattr(module, objectname)
    if provider_object is None:
        raise ImportError("Failed to import provider '{}'".format(provider_name))

    if provider_object == IBMQ:
        # register IBMQ first
        register_ibmq()

    try:
        # try as variable containing provider instance
        backend = provider_object.get_backend(backend_name)
    except:
        # try as provider class then
        provider_instance = provider_object()
        backend = provider_instance.get_backend(backend_name)

    if backend is None:
        raise ImportError("'{} not found in provider '{}'".format(backend_name, provider_object))

    return backend


def get_local_providers():
    providers = OrderedDict()
    # try Aer
    try:
        from qiskit import Aer
        providers['qiskit.Aer'] = [x.name() for x in Aer.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
    except Exception as e:
        logger.debug("Aer not loaded: '{}'.".format(str(e)))

    providers['qiskit.BasicAer'] = [x.name() for x in BasicAer.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
    return providers


def register_ibmq():
    # update registration info using internal methods because:
    # at this point I don't want to save to or remove credentials from disk
    # I want to update url, proxies etc without removing token and
    # re-adding in 2 methods

    try:
        credentials = None
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        if url is not None and url != '' and token is not None and token != '':
            credentials = Credentials(token,
                                      url,
                                      proxies=preferences.get_proxies({}))
        if credentials is not None:
            IBMQ._accounts[credentials.unique_id()] = IBMQSingleProvider(credentials, IBMQ)
            logger.debug("Registered with Qiskit IBMQ successfully.")
    except Exception as e:
        logger.debug("Failed to register with Qiskit IBMQ: {}".format(str(e)))


def get_ibmq_providers():
    # update registration info using internal methods because:
    # at this point I don't want to save to or remove credentials from disk
    # I want to update url, proxies etc without removing token and
    # re-adding in 2 methods

    providers = OrderedDict()
    try:
        register_ibmq()
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        providers['qiskit.IBMQ'] = [x.name() for x in IBMQ.backends(url=url, token=token) if x.name() not in _UNSUPPORTED_BACKENDS]
    except Exception as e:
        logger.debug("Failed to register with Qiskit IBMQ: {}".format(str(e)))

    return providers


def register_ibmq_and_get_known_providers():
    # update registration info using internal methods because:
    # at this point I don't want to save to or remove credentials from disk
    # I want to update url, proxies etc without removing token and
    # re-adding in 2 methods

    providers = get_local_providers()
    providers.update(get_ibmq_providers())
    return providers
