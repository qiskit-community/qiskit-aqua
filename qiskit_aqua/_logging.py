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
from collections import OrderedDict
from  qiskit_aqua import Preferences as AquaPreferences

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
    names = OrderedDict()
    names['qiskit_aqua'] = None
    preferences = AquaPreferences()
    packages = preferences.get_packages([])
    for package in packages:
        names[package] = None
        
    return list(names.keys())
        
def build_logging_config(level):
    """
     Creates a the configuration dict of the named loggers using the default SDK
     configuration provided by `_ALGO_LOGGING_CONFIG`:

    * console logging using a custom format for levels != level parameter.
    * console logging with simple format for level parameter.
    * set logger level to level parameter.
    """
    dict = copy.deepcopy(_ALGO_LOGGING_CONFIG)
    for name in _get_logging_names(): 
        dict['loggers'][name] = {   
                    'handlers' : ['h'],
                    'propagate' : False,
                    'level' : level
        }
    return dict
    
def get_logging_level():
    """get level for the named logger."""   
    return logging.getLogger('qiskit_aqua').getEffectiveLevel()

def set_logging_config(logging_config):
    """Update logger configurations using a SDK default one.

    Warning:
        This function modifies the configuration of the standard logging system
        for the loggers, and might interfere with custom logger
        configurations.
    """
    dictConfig(logging_config)
