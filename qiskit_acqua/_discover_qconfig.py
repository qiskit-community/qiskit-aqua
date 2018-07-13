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

"""
Methods for Qconfig.py module discovery and loading
"""

import importlib
import os
import logging

logger = logging.getLogger(__name__)

_QCONFIG_NAME = 'Qconfig'
_QCONFIG_FILE = _QCONFIG_NAME + '.py'

_QCONFIG_DISCOVERED = False
__QCONFIG__ = None

def _discover_qconfig_on_demand():
    """
    Attempts to discover Qconfog module, if not already discovered
    Returns:
        module: Qconfig module
    """
    global _QCONFIG_DISCOVERED
    global __QCONFIG__
    if not _QCONFIG_DISCOVERED:
        __QCONFIG__ = load_qconfig()
        if __QCONFIG__ is None:
            path = os.getcwd()
            __QCONFIG__ = discover_qconfig(path)
            if __QCONFIG__ is None:
                logger.debug('{} not loaded from {} and below'.format(_QCONFIG_NAME,path))

        _QCONFIG_DISCOVERED = True
        if __QCONFIG__ is not None:
            logger.debug('Loaded {} from {}'.format(_QCONFIG_NAME,os.path.abspath(__QCONFIG__.__file__)))

    return __QCONFIG__

def get_qconfig():
    """
    Returns current Qconfog module if found
    Returns:
        module: Qconfig module
    """
    return _discover_qconfig_on_demand()

def set_qconfig(qconfig):
    """
    Sets the Qconfig.py module.
    Args:
        qconfig (object): Qconfig module
    """
    global _QCONFIG_DISCOVERED
    global __QCONFIG__
    _QCONFIG_DISCOVERED = True
    __QCONFIG__ = qconfig

def load_qconfig():
    """
    Attemps to load the Qconfig.py searching the current environment.
    Returns:
        module: Qconfig module
    """

    try:
        modspec = importlib.util.find_spec(_QCONFIG_NAME)
        if modspec is not None:
            mod = importlib.util.module_from_spec(modspec)
            if mod is not None:
                modspec.loader.exec_module(mod)
                logger.debug('Loaded {}'.format(_QCONFIG_NAME))
                return mod
    except Exception as e:
        logger.debug('Failed to load {} error {}'.format(_QCONFIG_NAME, str(e)))
        return None

    return None

def discover_qconfig(directory):
    """
    Discovers the Qconfig.py and attempts to load it.
    Args:
        directory (str): Directory to search for Qconfig.py.
    Returns:
        module: Qconfig module
    """
    
    try:
        for item in os.listdir(directory):
            fullpath = os.path.join(directory,item)
            if item == _QCONFIG_FILE:
                try:
                    modspec = importlib.util.spec_from_file_location(_QCONFIG_NAME,fullpath)
                    if modspec is None: 
                        logger.debug('Failed to load {} {}'.format(_QCONFIG_NAME,fullpath))
                        return None

                    mod = importlib.util.module_from_spec(modspec)
                    modspec.loader.exec_module(mod)
                    return mod
                except Exception as e:
                    # Ignore if it could not be initialized.
                    logger.debug('Failed to load {} error {}'.format(fullpath, str(e)))
                    return None

            if item != '__pycache__' and not item.endswith('dSYM') and os.path.isdir(fullpath):
                mod = discover_qconfig(fullpath)
                if mod is not None:
                    return mod
                
    except Exception as e:
        # Ignore if it could not list
        logger.debug('Failed to list {} error {}'.format(directory, str(e)))
     
    return None
