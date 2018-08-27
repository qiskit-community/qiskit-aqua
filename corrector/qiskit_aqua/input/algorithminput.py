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


from abc import ABC, abstractmethod

from qiskit_aqua import AlgorithmError


class AlgorithmInput(ABC):

    _PROBLEM_SET = ['energy', 'excited_states', 'dynamics', 'search', 'svm_classification', 'ising']

    @abstractmethod
    def __init__(self, configuration=None):
        self._configuration = configuration
        if 'problems' not in configuration or len(configuration['problems']) <= 0:
            raise AlgorithmError('Algorithm Input missing or empty configuration problems')

        for problem in configuration['problems']:
            if problem not in AlgorithmInput._PROBLEM_SET:
                raise AlgorithmError('Problem {} not in known problem set {}'.format(problem, AlgorithmInput._PROBLEM_SET))

    @property
    def all_problems(self):
        return self._PROBLEM_SET.copy()

    @property
    def configuration(self):
        """
        Gets the configuration of this input form
        """
        return self._configuration

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
            params: A dictionary as originally created by to_params()
        """
        raise NotImplementedError()
