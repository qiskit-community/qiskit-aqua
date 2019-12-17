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

"""
This module implements the abstract base class for Pluggable modules.

To create add-on pluggable modules subclass the Pluggable
class in this module.
Doing so requires that the required pluggable interface is implemented.
"""

from abc import ABC, abstractmethod
import logging
import copy
import numpy as np
from qiskit.aqua import PluggableType
from qiskit.aqua.parser import JSONSchema


logger = logging.getLogger(__name__)


class Pluggable(ABC):
    """
    Base class for Pluggables.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    """

    CONFIGURATION = None

    # Configuration dictionary keys
    SECTION_KEY_ALGORITHM = PluggableType.ALGORITHM.value
    SECTION_KEY_OPTIMIZER = PluggableType.OPTIMIZER.value
    SECTION_KEY_VAR_FORM = PluggableType.VARIATIONAL_FORM.value
    SECTION_KEY_INITIAL_STATE = PluggableType.INITIAL_STATE.value
    SECTION_KEY_IQFT = PluggableType.IQFT.value
    SECTION_KEY_QFT = PluggableType.QFT.value
    SECTION_KEY_ORACLE = PluggableType.ORACLE.value
    SECTION_KEY_FEATURE_MAP = PluggableType.FEATURE_MAP.value
    SECTION_KEY_MULTICLASS_EXT = PluggableType.MULTICLASS_EXTENSION.value
    SECTION_KEY_UNCERTAINTY_PROBLEM = PluggableType.UNCERTAINTY_PROBLEM.value
    SECTION_KEY_UNIVARIATE_DIST = PluggableType.UNIVARIATE_DISTRIBUTION.value
    SECTION_KEY_MULTIVARIATE_DIST = PluggableType.MULTIVARIATE_DISTRIBUTION.value
    SECTION_KEY_EIGS = PluggableType.EIGENVALUES.value
    SECTION_KEY_RECIPROCAL = PluggableType.RECIPROCAL.value
    SECTION_KEY_DISCRIMINATIVE_NET = PluggableType.DISCRIMINATIVE_NETWORK.value
    SECTION_KEY_GENERATIVE_NETWORK = PluggableType.GENERATIVE_NETWORK.value

    @abstractmethod
    def __init__(self):
        self.check_pluggable_valid()
        self._configuration = copy.deepcopy(self.CONFIGURATION)

    @property
    def configuration(self):
        """Return pluggable configuration."""
        return self._configuration

    @staticmethod
    def check_pluggable_valid():
        """Checks if pluggable is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        """ validate input data """
        schema_dict = self.CONFIGURATION.get('input_schema')
        if schema_dict is None:
            return

        json_schema = JSONSchema(schema_dict)
        schema_property_names = json_schema.get_default_section_names()
        json_dict = {}
        for property_name in schema_property_names:
            if property_name in args_dict:
                value = args_dict[property_name]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                json_dict[property_name] = value

        json_schema.validate(json_dict)
