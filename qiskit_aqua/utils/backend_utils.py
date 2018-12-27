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

from qiskit import IBMQ
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


def get_aer_backend(backend_name):
    providers = ['qiskit.Aer', 'qiskit.BasicAer']
    for provider in providers:
        try:
            return get_backend_from_provider(provider, backend_name)
        except:
            pass

    raise ImportError("Backend '{}' not found in providers {}".format(backend_name, providers))


def get_backends_from_provider(provider_name):
    """
    Backends access method
    Args:
        provider_name (str): Fullname of provider instance global property or class
    Returns:
        list: backend names
    Raises:
        ImportError: Invalid provider name or failed to find provider
    """
    provider_object = _load_provider(provider_name)
    if provider_object == IBMQ:
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        kwargs = {}
        if url is not None and url != '':
            kwargs['url'] = url
        if token is not None and token != '':
            kwargs['token'] = token
        return [x.name() for x in provider_object.backends(**kwargs) if x.name() not in _UNSUPPORTED_BACKENDS]

    try:
        # try as variable containing provider instance
        return [x.name() for x in provider_object.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
    except:
        # try as provider class then
        try:
            provider_instance = provider_object()
            return [x.name() for x in provider_instance.backends() if x.name() not in _UNSUPPORTED_BACKENDS]
        except:
            pass

    raise ImportError("'Backends not found for provider '{}'".format(provider_object))


def get_backend_from_provider(provider_name, backend_name):
    """
    Backend access method
    Args:
        provider_name (str): Fullname of provider instance global property or class
        backend_name (str): name of backend for tgis provider
    Returns:
        object: backend object
    Raises:
        ImportError: Invalid provider name or failed to find provider
    """
    backend = None
    provider_object = _load_provider(provider_name)
    if provider_object == IBMQ:
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        kwargs = {}
        if url is not None and url != '':
            kwargs['url'] = url
        if token is not None and token != '':
            kwargs['token'] = token
        backend = provider_object.get_backend(backend_name, **kwargs)
    else:
        try:
            # try as variable containing provider instance
            backend = provider_object.get_backend(backend_name)
        except:
            # try as provider class then
            try:
                provider_instance = provider_object()
                backend = provider_instance.get_backend(backend_name)
            except:
                pass

    if backend is None:
        raise ImportError("'{} not found in provider '{}'".format(backend_name, provider_object))

    return backend


def get_local_providers():
    providers = OrderedDict()
    for provider in ['qiskit.Aer', 'qiskit.BasicAer']:
        try:
            providers[provider] = get_backends_from_provider(provider)
        except Exception as e:
            logger.debug("'{}' not loaded: '{}'.".format(provider, str(e)))

    return providers


def register_ibmq_and_get_known_providers():
    """Gets known local providers and registers IBMQ"""
    providers = get_local_providers()
    providers.update(_get_ibmq_provider())
    return providers


def get_provider_from_backend(backend_name):
    """
        Attempts to find a known provider that provides this backend
        Args:
            backend_name (str): name of backend for tgis provider
        Returns:
            str: provider name
        Raises:
            ImportError: Failed to find provider
    """
    providers = ['qiskit.Aer', 'qiskit.BasicAer', 'qiskit.IBMQ']
    for provider in providers:
        try:
            if get_backend_from_provider(provider, backend_name) is not None:
                return provider
        except:
            pass

    raise ImportError("Backend '{}' not found in providers {}".format(backend_name, providers))


def _load_provider(provider_name):
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
        # register IBMQ
        _register_ibmq()

    return provider_object


def _register_ibmq():
    """
    Update registration info using internal methods because:
    at this point I don't want to save to or remove credentials from disk
    I want to update url, proxies etc without removing token and
    re-adding in 2 methods
    """
    try:
        preferences = Preferences()
        url = preferences.get_url()
        token = preferences.get_token()
        if url is not None and url != '' and token is not None and token != '':
            found = False
            for account in IBMQ.active_accounts():
                if 'url' in account and 'token' in account and \
                        account['url'] == url and account['token'] == token:
                    found = True
                    break

            if not found:
                IBMQ.enable_account(token, url=url, proxies=preferences.get_proxies({}))
                logger.debug("Registered with Qiskit IBMQ successfully.")
    except Exception as e:
        logger.debug("Failed to register with Qiskit IBMQ: {}".format(str(e)))


def _get_ibmq_provider():
    """Registers IBMQ and return it"""
    providers = OrderedDict()
    try:
        providers['qiskit.IBMQ'] = get_backends_from_provider('qiskit.IBMQ')
    except Exception as e:
        logger.debug("Failed to register with Qiskit IBMQ: {}".format(str(e)))

    return providers
