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
            'format': '%(asctime)s:%(name)s:%(level_name)s: %(message)s'
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


def add_qiskit_aqua_logging_level(level_name, level_num, method_name=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `level_name` becomes an attribute of the `logging` module with the value
    `level_num`. `method_name` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `method_name` is not specified, `level_name.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> add_qiskit_aqua_logging_level('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(
            '{} already defined in logging module'.format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError(
            '{} already defined in logging module'.format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError(
            '{} already defined in logger class'.format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, *args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)
