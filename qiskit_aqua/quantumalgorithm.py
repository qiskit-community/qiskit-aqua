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
import sys
import functools

import numpy as np
from qiskit import __version__ as qiskit_version
from qiskit.wrapper import register as q_register
from qiskit.wrapper import unregister as q_unregister
from qiskit.wrapper import registered_providers as q_registered_providers
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper import execute as q_execute
from qiskit.wrapper import available_backends, get_backend

from qiskit_aqua import get_qconfig, AlgorithmError
from qiskit_aqua.utils import summarize_circuits

logger = logging.getLogger(__name__)


class QuantumAlgorithm(ABC):

    # Configuration dictionary keys
    SECTION_KEY_ALGORITHM = 'algorithm'
    SECTION_KEY_OPTIMIZER = 'optimizer'
    SECTION_KEY_VAR_FORM = 'variational_form'
    SECTION_KEY_INITIAL_STATE = 'initial_state'
    SECTION_KEY_IQFT = 'iqft'
    SECTION_KEY_ORACLE = 'oracle'

    MAX_CIRCUITS_PER_JOB = 300

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

    def setup_quantum_backend(self, backend='local_statevector_simulator', shots=1024, skip_transpiler=False,
                              timeout=None, wait=5, noise_params=None):
        """
        Setup the quantum backend.

        Args:
            backend (str): name of selected backend
            shots (int): number of shots for the backend
            skip_transpiler (bool): flag of skipping gate mapping, be aware that only
                                     basis gates of the backend are executed others are skipped.
            timeout (float or None): seconds to wait for job. If None, wait indefinitely.
            wait (float): seconds between queries
            noise_params (dict): the noise setting for simulator

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

        shots = 1 if 'statevector' in backend else shots
        noise_params = noise_params if 'simulator' in backend else None

        if backend.startswith('local'):
            self._qjob_config.pop('wait', None)
            self.MAX_CIRCUITS_PER_JOB = sys.maxsize

        my_backend = get_backend(backend)
        self._execute_config = {'shots': shots,
                                'skip_transpiler': skip_transpiler,
                                'config': {"noise_params": noise_params},
                                'basis_gates': my_backend.configuration['basis_gates'],
                                'coupling_map': my_backend.configuration['coupling_map'],
                                'initial_layout': None,
                                'max_credits': 10,
                                'seed': self._random_seed,
                                'qobj_id': None,
                                'hpc': None}

        info = "Algorithm: '{}' setup with backend '{}', with following setting:\n {}\n{}".format(
            self._configuration['name'], my_backend.configuration['name'], self._execute_config, self._qjob_config)

        logger.info('Qiskit Terra version {}'.format(qiskit_version))
        logger.info(info)

    def execute(self, circuits):
        """
        A wrapper for all algorithms to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute

        Returns:
            Result or [Result]: Result objects it will be a list if number of circuits
            exceed the maximum number (300)
        """

        if not isinstance(circuits, list):
            circuits = [circuits]
        jobs = []
        chunks = int(np.ceil(len(circuits) / self.MAX_CIRCUITS_PER_JOB))
        for i in range(chunks):
            sub_circuits = circuits[i*self.MAX_CIRCUITS_PER_JOB:(i+1)*self.MAX_CIRCUITS_PER_JOB]
            jobs.append(q_execute(sub_circuits, self._backend, **self._execute_config))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(summarize_circuits(circuits))

        results = []
        for job in jobs:
            results.append(job.result(**self._qjob_config))

        result = functools.reduce(lambda x, y: x + y, results)
        return result

    @staticmethod
    def register_and_get_operational_backends(qconfig):
        try:
            for provider in q_registered_providers():
                if isinstance(provider,IBMQProvider):
                    q_unregister(provider)
                    logger.debug("Provider 'IBMQProvider' unregistered with Qiskit successfully.")
                    break
        except Exception as e:
                logger.debug("Failed to unregister provider 'IBMQProvider' with Qiskit: {}".format(str(e)))

        if qconfig is not None:
            hub = qconfig.config.get('hub', None)
            group = qconfig.config.get('group', None)
            project = qconfig.config.get('project', None)
            proxies = qconfig.config.get('proxies', None)
            verify = qconfig.config.get('verify', True)
            try:
                q_register(qconfig.APItoken,
                           provider_class=IBMQProvider,
                           url=qconfig.config["url"],
                           hub=hub,
                           group=group,
                           project=project,
                           proxies=proxies,
                           verify=verify)
                logger.debug("Provider 'IBMQProvider' registered with Qiskit successfully.")
            except Exception as e:
                logger.debug("Failed to register provider 'IBMQProvider' with Qiskit: {}".format(str(e)))

        backends = available_backends()
        backends = [x for x in backends if x not in QuantumAlgorithm.UNSUPPORTED_BACKENDS]
        return backends

    @abstractmethod
    def init_params(self, params, algo_input):
        pass

    @abstractmethod
    def run(self):
        pass
