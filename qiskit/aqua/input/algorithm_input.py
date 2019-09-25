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

""" Algorithm Input """

import copy
from abc import abstractmethod
from qiskit.aqua import Pluggable
from qiskit.aqua import AquaError


class AlgorithmInput(Pluggable):
    """ Algorithm Input """
    _PROBLEM_SET = ['energy', 'excited_states', 'eoh', 'classification', 'ising', 'linear_system',
                    'distribution_learning_loading']

    @abstractmethod
    def __init__(self):
        super().__init__()
        if 'problems' not in self.configuration or not self.configuration['problems']:
            raise AquaError('Algorithm Input missing or empty configuration problems')

        for problem in self.configuration['problems']:
            if problem not in AlgorithmInput._PROBLEM_SET:
                raise AquaError(
                    'Problem {} not in known problem set {}'.format(problem,
                                                                    AlgorithmInput._PROBLEM_SET))

    @property
    def all_problems(self):
        """ returns all problems """
        return copy.deepcopy(self._PROBLEM_SET)

    @property
    def problems(self):
        """
        Gets the set of problems that this input form supports
        """
        return self.configuration.problems

    @abstractmethod
    def to_params(self):
        """
        Convert the derived algorithminput class fields to a dictionary where the values are in a
        form that can be saved to json
        Returns:
            Dictionary of input fields
        """
        raise NotImplementedError()

    @abstractmethod
    def from_params(self, params):
        """
        Load the dictionary into the algorithminput class fields. This dictionary being that as
        created by to_params()
        Args:
            params (dict): A dictionary as originally created by to_params()
        """
        raise NotImplementedError()
