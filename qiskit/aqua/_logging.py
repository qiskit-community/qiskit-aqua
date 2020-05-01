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
"""Utilities for logging."""

import os
import copy
import logging
from logging.config import dictConfig
from collections import OrderedDict
import pkg_resources

_ALGO_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'f': {
            'format': '%(asctime)s:%(name)s:%(levelname)s: %(message)s'
        },
    },
    'handlers': {
        'h': {
            'class': 'logging.StreamHandler',
            'formatter': 'f'
        }
    },
    'loggers': {}
}


def _get_logging_names():
    # pylint: disable=import-outside-toplevel
    from qiskit.aqua import PLUGGABLES_ENTRY_POINT
    names = OrderedDict()
    names['qiskit.aqua'] = None
    for entry_point in pkg_resources.iter_entry_points(PLUGGABLES_ENTRY_POINT):
        names[entry_point.module_name] = None

    return list(names.keys())


def build_logging_config(level, filepath=None):
    """
     Creates a the configuration dict of the named loggers using the default SDK
     configuration provided by `_ALGO_LOGGING_CONFIG`:

    * console logging using a custom format for levels != level parameter.
    * console logging with simple format for level parameter.
    * set logger level to level parameter.

    Args:
        level (number): logging level
        filepath (str): file to receive logging data
    Returns:
        dict: New configuration dictionary
    """
    dict_conf = copy.deepcopy(_ALGO_LOGGING_CONFIG)
    if filepath is not None:
        filepath = os.path.expanduser(filepath)
        dict_conf['handlers']['f'] = {
            'class': 'logging.FileHandler',
            'formatter': 'f',
            'filename': filepath,
            'mode': 'w'
        }

    handlers = list(dict_conf['handlers'].keys())
    for name in _get_logging_names():
        dict_conf['loggers'][name] = {
            'handlers': handlers,
            'propagate': False,
            'level': level
        }
    return dict_conf


def get_logging_level():
    """get level for the named logger."""
    return logging.getLogger('qiskit.aqua').getEffectiveLevel()


def set_logging_config(logging_config):
    """Update logger configurations using a SDK default one.

    Warning:
        This function modifies the configuration of the standard logging system
        for the loggers, and might interfere with custom logger
        configurations.
    """
    dictConfig(logging_config)


def get_qiskit_aqua_logging():
    """
    Returns the current Aqua logging level

    Returns:
        int: logging level
    """
    return get_logging_level()


def set_qiskit_aqua_logging(level, filepath=None):
    """
    Updates the Aqua logging with the appropriate logging level

    Args:
        level (number): logging level
        filepath (str): file to receive logging data
    """
    set_logging_config(build_logging_config(level, filepath))
