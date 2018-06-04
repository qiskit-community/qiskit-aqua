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
This module implements the abstract base class for algorithm modules.

To create add-on algorithm modules subclass the QuantumAlgorithm class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import ABC, abstractmethod
import logging

import numpy as np
from qiskit.wrapper import register as q_register
from qiskit.wrapper import execute as q_execute
from qiskit.wrapper import available_backends, get_backend

from qiskit_acqua import get_qconfig, AlgorithmError

logger = logging.getLogger(__name__)


class QuantumAlgorithm(ABC):

    # Configuration dictionary keys
    SECTION_KEY_ALGORITHM = 'algorithm'
    SECTION_KEY_OPTIMIZER = 'optimizer'
    SECTION_KEY_VAR_FORM = 'variational_form'
    SECTION_KEY_INITIAL_STATE = 'initial_state'
    SECTION_KEY_IQFT = 'iqft'
    SECTION_KEY_ORACLE = 'oracle'

    UNSUPPORTED_BACKENDS = ['local_unitary_simulator', 'local_clifford_simulator']

    EQUIVALENT_BACKENDS = {'local_statevector_simulator_py': 'local_statevector_simulator',
                           'local_statevector_simulator_cpp': 'local_statevector_simulator',
                           'local_statevector_simulator_sympy': 'local_statevector_simulator',
                           'local_statevector_simulator_projectq': 'local_statevector_simulator',
                           'local_qasm_simulator_py': 'local_qasm_simulator',
                           'local_qasm_simulator_cpp': 'local_qasm_simulator',
                           'local_qasm_simulator_projectq': 'local_qasm_simulator'
                           }
    """
    Base class for Algorithms.

    This method should initialize the module and its configuration, and
    use an exception if a component of the module is available.

    Args:
        configuration (dict): configuration dictionary
    """
    @abstractmethod
    def __init__(self, configuration=None):
        self._configuration = configuration
        self._qconfig = None
        self._backend = None
        self._execute_config = {}
        self._qjob_config = {}
        self._random_seed = None
        self._random = None

    @property
    def configuration(self):
        """Return algorithm configuration"""
        return self._configuration

    @property
    def qconfig(self):
        """Return Qconfig configuration"""
        if self._qconfig is None:
            self._qconfig = get_qconfig()

        return self._qconfig

    @qconfig.setter
    def qconfig(self, new_qconfig):
        """Sets Qconfig configuration"""
        self._qconfig = new_qconfig

    @property
    def random_seed(self):
        """Return random seed"""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Sets random seed"""
        self._random_seed = seed

    @property
    def random(self):
        """Return a numpy random"""
        if self._random is None:
            if self._random_seed is None:
                self._random = np.random
            else:
                self._random = np.random.RandomState(self._random_seed)
        return self._random

    def setup_quantum_backend(self, backend='local_statevector_simulator', shots=1024, skip_translation=False,
                              timeout=None, wait=5, noise_config=None):
        """
        Setup the quantum backend.

        Args:
            backend (str): name of selected backend
            shots (int): number of shots for the backend
            skip_translation (bool): flag of skipping gate mapping, be aware that only
                                     basis gates of the backend are executed others are skipped.
            timeout (float or None): seconds to wait for job. If None, wait indefinitely.
            wait (float): seconds between queries
            noise_config (dict): the noise setting for simulator

        Raises:
            AlgorithmError: set backend with invalid Qconfig
        """
        operational_backends = self.register_and_get_operational_backends(self.qconfig)
        if self.EQUIVALENT_BACKENDS.get(backend, backend) not in operational_backends:
            raise AlgorithmError("This backend '{}' is not operational for the quantum algorithm\
                , please check your Qconfig.py, or select any one below: {}".format(backend, operational_backends))

        self._backend = backend
        self._qjob_config = {'timeout': timeout,
                             'wait': wait}

        shots = 1 if 'statevector' in backend else 1024 if shots == 1 else shots
        noise_config = noise_config if 'simulator' in backend else None

        if backend.startswith('local'):
            self._qjob_config.pop('wait', None)

        my_backend = get_backend(backend)
        self._execute_config = {'shots': shots,
                                'skip_translation': skip_translation,
                                'config': noise_config,
                                'basis_gates': my_backend.configuration['basis_gates'],
                                'coupling_map': my_backend.configuration['coupling_map'],
                                'initial_layout': None,
                                'max_credits': 10,
                                'seed': self._random_seed,
                                'qobj_id': None,
                                'hpc': None}

        info = "Algorithm: '{}' setup with backend '{}', with following setting:\n {}\n{}".format(
            self._configuration['name'], my_backend.configuration['name'], self._execute_config, self._qjob_config)

        logger.info(info)

    def execute(self, circuits):
        """
        A wrapper for all algorithms to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        """
        job = q_execute(circuits, self._backend, **self._execute_config)
        result = job.result(**self._qjob_config)
        return result

    @staticmethod
    def register_and_get_operational_backends(qconfig):
        if qconfig is not None:
            hub = qconfig.config.get('hub', None)
            group = qconfig.config.get('group', None)
            project = qconfig.config.get('project', None)
            try:
                q_register(qconfig.APItoken, qconfig.config["url"], hub=hub, group=group, project=project)
            except:
                logger.debug("WARNING: Can not register quantum backends. Check your Qconfig.py.")

        backends = available_backends()
        backends = [x for x in backends if x not in QuantumAlgorithm.UNSUPPORTED_BACKENDS]
        return backends

    @abstractmethod
    def init_params(self, params, algo_input):
        pass

    @abstractmethod
    def run(self):
        pass
