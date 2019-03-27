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

import numpy as np
from .aqua_error import AquaError
from qiskit.util import local_hardware_info
import qiskit
import logging

logger = logging.getLogger(__name__)


class QiskitAquaGlobals(object):
    """Aqua class for global properties."""

    CPU_COUNT = local_hardware_info()['cpus']

    def __init__(self):
        self._random_seed = None
        self._num_processes = QiskitAquaGlobals.CPU_COUNT
        self._random = None

    @property
    def random_seed(self):
        """Return random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Set random seed."""
        self._random_seed = seed
        self._random = None

    @property
    def num_processes(self):
        """Return num processes."""
        return self._num_processes

    @num_processes.setter
    def num_processes(self, num_processes):
        """Set num processes."""
        if num_processes < 1:
            raise AquaError('Invalid Number of Processes {}.'.format(num_processes))
        if num_processes > QiskitAquaGlobals.CPU_COUNT:
            raise AquaError('Number of Processes {} cannot be greater than cpu count {}.'.format(num_processes, QiskitAquaGlobals._CPU_COUNT))
        self._num_processes = num_processes
        # TODO: change Terra CPU_COUNT until issue gets resolved: https://github.com/Qiskit/qiskit-terra/issues/1963
        try:
            qiskit.tools.parallel.CPU_COUNT = self.num_processes
        except Exception as e:
            logger.warning("Failed to set qiskit.tools.parallel.CPU_COUNT to value: '{}': Error: '{}'".format(self.num_processes, str(e)))

    @property
    def random(self):
        """Return a numpy random."""
        if self._random is None:
            if self._random_seed is None:
                self._random = np.random
            else:
                self._random = np.random.RandomState(self._random_seed)
        return self._random


# Global instance to be used as the entry point for globals.
aqua_globals = QiskitAquaGlobals()
