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
This module contains the definition of a base class for a chemistry operator.
Such an operator takes a QMolecule and produces an input for
a quantum algorithm
"""
from abc import ABC, abstractmethod
import logging
import copy
import numpy as np
import jsonschema
from qiskit.chemistry import QiskitChemistryError

logger = logging.getLogger(__name__)


class ChemistryOperator(ABC):
    """
    Base class for ChemistryOperator.

        This method should initialize the module and its configuration, and
        use an exception if a component of the module is
        available.
    """

    CONFIGURATION = None
    INFO_NUM_PARTICLES = 'num_particles'
    INFO_NUM_ORBITALS = 'num_orbitals'
    INFO_TWO_QUBIT_REDUCTION = 'two_qubit_reduction'

    @abstractmethod
    def __init__(self):
        self._configuration = copy.deepcopy(self.CONFIGURATION)
        self._molecule_info = {}

    @property
    def configuration(self):
        """ returns configuration """
        return self._configuration

    @staticmethod
    def check_chemistry_operator_valid():
        """Checks if Chemistry Operator is ready for use. Throws an exception if not"""
        pass

    def validate(self, args_dict):
        """ validate input """
        schema_dict = self.CONFIGURATION.get('input_schema', None)
        if schema_dict is None:
            return

        properties_dict = schema_dict.get('properties', None)
        if properties_dict is None:
            return

        json_dict = {}
        for property_name, _ in properties_dict.items():
            if property_name in args_dict:
                value = args_dict[property_name]
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                json_dict[property_name] = value
        try:
            jsonschema.validate(json_dict, schema_dict)
        except jsonschema.exceptions.ValidationError as vex:
            raise QiskitChemistryError(vex.message)

    @abstractmethod
    def run(self, qmolecule):
        """
        Convert the qmolecule, according to the ChemistryOperator, into an Operator
        that can be given to a QuantumAlgorithm

        Args:
            qmolecule (QMolecule): from a chemistry driver

        Returns:
            Tuple: (qubit_op, aux_ops)
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
            Tuple: (lines, result) Final computation result
        """
        lines, result = self._process_algorithm_result(algo_result)
        result['algorithm_retvals'] = algo_result
        return lines, result

    @abstractmethod
    def _process_algorithm_result(self, algo_result):
        raise NotImplementedError

    @property
    def molecule_info(self):
        """ returns molecule info """
        return self._molecule_info

    def _add_molecule_info(self, key, value):
        self._molecule_info[key] = value
