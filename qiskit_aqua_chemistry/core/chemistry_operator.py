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
    def __init__(self, configuration=None):
        self._configuration = configuration
        self._molecule_info = {}
        pass

    @property
    def configuration(self):
        return self._configuration

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
