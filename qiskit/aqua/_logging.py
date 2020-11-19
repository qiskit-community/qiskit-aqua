# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for logging."""

from typing import Optional, Dict, List, Any
import os
import logging
from enum import Enum
from logging.config import dictConfig


class QiskitLogDomains(Enum):
    """ Qiskit available Log Domains  """
    DOMAIN_AQUA = 'qiskit.aqua'
    DOMAIN_CHEMISTRY = 'qiskit.chemistry'
    DOMAIN_FINANCE = 'qiskit.finance'
    DOMAIN_ML = 'qiskit.ml'
    DOMAIN_OPTIMIZATION = 'qiskit.optimization'


def build_logging_config(level: int,
                         domains: List[QiskitLogDomains],
                         filepath: Optional[str] = None) -> Dict:
    """
    Creates a configuration dict for the given domains

    * console logging using a custom format for levels != level parameter.
    * console logging with simple format for level parameter.
    * set logger level to level parameter.

    Args:
        level: logging level
        domains: Qiskit domains to be logged
        filepath: file to receive logging data
    Returns:
        dict: New configuration dictionary
    """
    dict_conf = {
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
    }  # type: Dict[str, Any]
    if filepath is not None:
        filepath = os.path.expanduser(filepath)
        dict_conf['handlers']['f'] = {
            'class': 'logging.FileHandler',
            'formatter': 'f',
            'filename': filepath,
            'mode': 'w'
        }

    handlers = list(dict_conf['handlers'].keys())
    for domain in domains:
        dict_conf['loggers'][domain.value] = {
            'handlers': handlers,
            'propagate': False,
            'level': level
        }
    return dict_conf


def get_logging_level(domain: QiskitLogDomains) -> int:
    """
    Get level for the named logger.
    Args:
        domain: Qiskit domain.
    Returns:
        int: logging level
    """
    return logging.getLogger(domain.value).getEffectiveLevel()


def set_logging_level(level: int,
                      domains: Optional[List[QiskitLogDomains]],
                      filepath: Optional[str] = None) -> None:
    """
    Updates given domains with the appropriate logging level

    Args:
        level: logging level
        domains: Qiskit domains to be logged.
        filepath: file to receive logging data
    """
    set_logging_config(build_logging_config(level, domains, filepath))


def set_logging_config(logging_config: Dict) -> None:
    """Update logger configurations.

    Warning:
        This function modifies the configuration of the standard logging system
        for all loggers, and might interfere with custom logger
        configurations.
    """
    dictConfig(logging_config)


def get_qiskit_aqua_logging() -> int:
    """
    Returns the current Aqua logging level

    Returns:
        int: logging level
    """
    return get_logging_level(QiskitLogDomains.DOMAIN_AQUA)


def set_qiskit_aqua_logging(level: int, filepath: Optional[str] = None) -> None:
    """
    Updates the Qiskit Aqua logging with the appropriate logging level

    Args:
        level: logging level
        filepath: file to receive logging data
    """
    set_logging_level(level, [QiskitLogDomains.DOMAIN_AQUA], filepath)
