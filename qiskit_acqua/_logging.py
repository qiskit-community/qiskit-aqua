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
"""Utilities for logging."""

import copy
import logging
from logging.config import dictConfig

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
        
def build_logging_config(names,level):
    """
     Creates a the configuration dict of the named loggers using the default SDK
     configuration provided by `_ALGO_LOGGING_CONFIG`:

    * console logging using a custom format for levels != level parameter.
    * console logging with simple format for level parameter.
    * set logger level to level parameter.
    """
    dict = copy.deepcopy(_ALGO_LOGGING_CONFIG)
    for name in names: 
        dict['loggers'][name] = {   
                    'handlers' : ['h'],
                    'propagate' : False,
                    'level' : level
        }
    return dict

def get_logger_levels_for_names(names):
    """get levels for the named loggers."""
    return[logging.getLogger(name).getEffectiveLevel() for name in names]

def set_logger_config(logging_config):
    """Update logger configurations using a SDK default one.

    Warning:
        This function modifies the configuration of the standard logging system
        for the loggers, and might interfere with custom logger
        configurations.
    """
    dictConfig(logging_config)

def unset_loggers(names):
    """Remove the handlers for the named loggers."""
    for name in names: 
        name_logger = logging.getLogger(name)
        if name_logger is not None:
            for handler in name_logger.handlers:
                name_logger.removeHandler(handler)
