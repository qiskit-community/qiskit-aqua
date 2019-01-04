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
This module contains the definition of a base class for a chemistry operator.
Such an operator takes a QMolecule and produces an input for
a quantum algorithm
"""
from abc import ABC, abstractmethod
import logging
import copy
from qiskit_aqua.parser import JSONSchema

logger = logging.getLogger(__name__)


class ChemistryOperator(ABC):
    """
    Base class for ChemistryOperator.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.

        Args:
            configuration (dict): configuration dictionary
    """

    INFO_NUM_PARTICLES = 'num_particles'
    INFO_NUM_ORBITALS = 'num_orbitals'
    INFO_TWO_QUBIT_REDUCTION = 'two_qubit_reduction'

    @abstractmethod
    def __init__(self):
        self._configuration = copy.deepcopy(self.CONFIGURATION)
        self._molecule_info = {}

    @property
    def configuration(self):
        return self._configuration

    @staticmethod
    def check_chemistry_operator_valid():
        """Checks if Chemistry Operator is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        jsonSchema = JSONSchema(schema_dict)
        schema_property_names = jsonSchema.get_default_section_names()
        json_dict = {}
        for property_name in schema_property_names:
            if property_name in args_dict:
                json_dict[property_name] = args_dict[property_name]

        jsonSchema.validate(json_dict)

    @classmethod
    def init_params(cls, params):
        """
        Initialize via parameters dictionary.

        Args:
            params (dict): parameters dictionary

        Returns:
            Chemistry Operator: Chemistry Operator object
        """
        kwargs = {k: v for k, v in params.items() if k != 'name'}
        logger.debug('init_params: {}'.format(kwargs))
        return cls(**kwargs)

    @abstractmethod
    def run(self, qmolecule):
        """
        Convert the qmolecule, according to the ChemistryOperator, into an Operator
        that can be given to a QuantumAlgorithm

        Args:
            qmolecule: QMolecule from a chemistry driver

        Returns:
            Algorithm input class instance
        """
        raise NotImplementedError

    def process_algorithm_result(self, algo_result):
        """
        Takes the algorithm result and processes it as required, e.g. by
        combination of any parts that were classically computed, for the
        final result.

        Args:
            algo_result (dict): Result from algorithm

        Returns:
            Final computation result
        """
        lines, result = self._process_algorithm_result(algo_result)
        result['algorithm_retvals'] = algo_result
        return lines, result

    @abstractmethod
    def _process_algorithm_result(self, algo_result):
        raise NotImplementedError

    @property
    def molecule_info(self):
        return self._molecule_info

    def _add_molecule_info(self, key, value):
        self._molecule_info[key] = value
