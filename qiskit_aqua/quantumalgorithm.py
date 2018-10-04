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

To create add-on algorithm modules subclass the QuantumAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import ABC, abstractmethod
import logging
import sys

import numpy as np
import qiskit
from qiskit import __version__ as qiskit_version
from qiskit.backends.ibmq.credentials import Credentials
from qiskit.backends.ibmq.ibmqsingleprovider import IBMQSingleProvider
from qiskit_aqua import AlgorithmError
from qiskit_aqua.utils import run_circuits
from qiskit_aqua import Preferences

logger = logging.getLogger(__name__)


class QuantumAlgorithm(ABC):

    # Configuration dictionary keys
    SECTION_KEY_ALGORITHM = 'algorithm'
    SECTION_KEY_OPTIMIZER = 'optimizer'
    SECTION_KEY_VAR_FORM = 'variational_form'
    SECTION_KEY_INITIAL_STATE = 'initial_state'
    SECTION_KEY_IQFT = 'iqft'
    SECTION_KEY_ORACLE = 'oracle'
    SECTION_KEY_FEATURE_MAP = 'feature_map'
    SECTION_KEY_MULTICLASS_EXTENSION = 'multiclass_extension'

    MAX_CIRCUITS_PER_JOB = 300

    UNSUPPORTED_BACKENDS = [
        'unitary_simulator', 'clifford_simulator']

    EQUIVALENT_BACKENDS = {'statevector_simulator_py': 'statevector_simulator',
                           'statevector_simulator_sympy': 'statevector_simulator',
                           'statevector_simulator_projectq': 'statevector_simulator',
                           'qasm_simulator_py': 'qasm_simulator',
                           'qasm_simulator_projectq': 'qasm_simulator'
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
        self._backend = None
        self._execute_config = {}
        self._qjob_config = {}
        self._random_seed = None
        self._random = None
        self._show_circuit_summary = False

    @property
    def configuration(self):
        """Return algorithm configuration"""
        return self._configuration

    @property
    def random_seed(self):
        """Return random seed"""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed):
        """Set random seed"""
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

    @property
    def backend(self):
        """Return backend"""
        return self._backend

    def enable_circuit_summary(self):
        """Enable showing the summary of circuits"""
        self._show_circuit_summary = True

    def disable_circuit_summary(self):
        """Disable showing the summary of circuits"""
        self._show_circuit_summary = False

    def setup_quantum_backend(self, backend='statevector_simulator', shots=1024, skip_transpiler=False,
                              noise_params=None, coupling_map=None, initial_layout=None, hpc_params=None,
                              basis_gates=None, max_credits=10, timeout=None, wait=5):
        """
        Setup the quantum backend.

        Args:
            backend (str): name of selected backend
            shots (int): number of shots for the backend
            skip_transpiler (bool): skip most of the compile steps and produce qobj directly
            noise_params (dict): the noise setting for simulator
            coupling_map (list): coupling map (perhaps custom) to target in mapping
            initial_layout (dict): initial layout of qubits in mapping
            hpc_params (dict): HPC simulator parameters
            basis_gates (str): comma-separated basis gate set to compile to
            max_credits (int): maximum credits to use
            timeout (float or None): seconds to wait for job. If None, wait indefinitely.
            wait (float): seconds between queries

        Raises:
            AlgorithmError: set backend with invalid Qconfig
        """
        operational_backends = self.register_and_get_operational_backends()
        if QuantumAlgorithm.EQUIVALENT_BACKENDS.get(backend, backend) not in operational_backends:
            raise AlgorithmError("This backend '{}' is not operational for the quantum algorithm, \
                                 select any one below: {}".format(backend, operational_backends))

        self._backend = backend
        self._qjob_config = {'timeout': timeout,
                             'wait': wait}

        shots = 1 if 'statevector' in backend else shots
        noise_params = noise_params if 'simulator' in backend else None

        my_backend = None
        try:
            my_backend = qiskit.Aer.get_backend(backend)
            self._qjob_config.pop('wait', None)
            self.MAX_CIRCUITS_PER_JOB = sys.maxsize
        except KeyError:
            my_backend = qiskit.IBMQ.get_backend(backend)

        if coupling_map is None:
            coupling_map = my_backend.configuration()['coupling_map']
        if basis_gates is None:
            basis_gates = my_backend.configuration()['basis_gates']

        self._execute_config = {'shots': shots,
                                'skip_transpiler': skip_transpiler,
                                'config': {"noise_params": noise_params},
                                'basis_gates': basis_gates,
                                'coupling_map': coupling_map,
                                'initial_layout': initial_layout,
                                'max_credits': max_credits,
                                'seed': self._random_seed,
                                'qobj_id': None,
                                'hpc': hpc_params}

        info = "Algorithm: '{}' setup with backend '{}', with following setting:\n {}\n{}".format(
            self._configuration['name'], my_backend.configuration()['name'], self._execute_config, self._qjob_config)

        logger.info('Qiskit Terra version {}'.format(qiskit_version))
        logger.info(info)

    def execute(self, circuits):
        """
        A wrapper for all algorithms to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute

        Returns:
            Result: Result object
        """
        result = run_circuits(circuits, self._backend, self._execute_config,
                              self._qjob_config, max_circuits_per_job=self.MAX_CIRCUITS_PER_JOB,
                              show_circuit_summary=self._show_circuit_summary)
        if self._show_circuit_summary:
            self.disable_circuit_summary()

        return result

    @staticmethod
    def register_and_get_operational_backends(token=None, url=None, **kwargs):
        # update registration info using internal methods because:
        # at this point I don't want to save to or removecredentials from disk
        # I want to update url, proxies etc without removing token and
        # re-adding in 2 methods

        try:
            credentials = None
            if token is not None:
                credentials = Credentials(token, url, **kwargs)
            else:
                preferences = Preferences()
                if preferences.get_token() is not None:
                    credentials = Credentials(preferences.get_token(),
                                              preferences.get_url(),
                                              proxies=preferences.get_proxies({}))
            if credentials is not None:
                qiskit.IBMQ._accounts[credentials.unique_id()] = IBMQSingleProvider(
                    credentials, qiskit.IBMQ)
                logger.debug("Registered with Qiskit successfully.")
        except Exception as e:
            logger.debug(
                "Failed to register with Qiskit: {}".format(str(e)))

        backends = set()
        local_backends = [x.name() for x in qiskit.Aer.backends()]
        for local_backend in local_backends:
            backend = None
            for group_name, names in qiskit.Aer.grouped_backend_names().items():
                if local_backend in names:
                    backend = group_name
                    break
            if backend is None:
                backend = local_backend

            supported = True
            for unsupported_backend in QuantumAlgorithm.UNSUPPORTED_BACKENDS:
                if backend.startswith(unsupported_backend):
                    supported = False
                    break

            if supported:
                backends.add(backend)

        return list(backends) + [x.name() for x in qiskit.IBMQ.backends()]

    @abstractmethod
    def init_params(self, params, algo_input):
        pass

    @abstractmethod
    def run(self):
        pass
