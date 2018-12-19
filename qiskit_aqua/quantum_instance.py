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

import logging
from qiskit import __version__ as terra_version
from qiskit import IBMQ, BasicAer
from qiskit.providers.ibmq.credentials import Credentials
from qiskit.providers.ibmq.ibmqsingleprovider import IBMQSingleProvider

from qiskit_aqua_cmd import Preferences
from qiskit_aqua.utils import compile_and_run_circuits
from qiskit_aqua import get_aer_backend, get_aer_backends

logger = logging.getLogger(__name__)


class QuantumInstance:
    """Quantum Backend including execution setting."""

    UNSUPPORTED_BACKENDS = [
        'unitary_simulator', 'clifford_simulator']

    EQUIVALENT_BACKENDS = {'statevector_simulator_py': 'statevector_simulator',
                           'statevector_simulator_sympy': 'statevector_simulator',
                           'statevector_simulator_projectq': 'statevector_simulator',
                           'qasm_simulator_py': 'qasm_simulator',
                           'qasm_simulator_projectq': 'qasm_simulator'
                           }

    BACKEND_CONFIG = ['basis_gates', 'config', 'coupling_map', 'seed', 'memory']
    COMPILE_CONFIG = ['pass_manager', 'initial_layout', 'seed_mapper', 'qobj_id']
    RUN_CONFIG = ['shots', 'max_credits']
    QJOB_CONFIG = ['timeout', 'wait']

    #  based on table at https://github.com/Qiskit/qiskit-terra/tree/master/src/qasm-simulator-cpp#table-of-config-options
    SIMULATOR_CONFIG = ["shots_threads", "data", "noise_params", "initial_state",
                        "target_states", "renom_target_states", "chop",
                        "max_memory", "max_threads_shot", "max_threads_gate", "threshold_omp_gate"]

    def __init__(self, backend, shots=1024, max_credits=10, config=None, seed=None,
                 initial_layout=None, pass_manager=None, seed_mapper=None, memory=False,
                 timeout=None, wait=5):
        """Constructor.

        Args:
            backend (BaseBackend): instance of selected backend
            shots (int): number of shots for the backend
            max_credits (int): maximum credits to use
            config (dict): all config setting for simulator
            seed (int): the random seed for simulator
            initial_layout (dict): initial layout of qubits in mapping
            pass_manager (PassManager): pass manager to handle how to compile the circuits
            seed_mapper (int): the random seed for circuit mapper
            memory (bool): if True, per-shot measurement bitstrings are returned as well
            timeout (float or None): seconds to wait for job. If None, wait indefinitely.
            wait (float): seconds between queries to result
        """
        self._backend = backend

        if self.is_statevector and shots != 1:
            logger.info("statevector backend only works with shot=1, change "
                        "shots from {} to 1.".format(shots))
            shots = 1

        coupling_map = getattr(backend.configuration(), 'coupling_map', None)
        # TODO: basis gates will be [str] rather than str
        basis_gates = backend.configuration().basis_gates
        basis_gates = ','.join(basis_gates)

        self._compile_config = {
            'pass_manager': pass_manager,
            'initial_layout': initial_layout,
            'seed_mapper': seed_mapper,
            'qobj_id': None
        }

        self._run_config = {
            'shots': shots,
            'max_credits': max_credits
        }

        self._backend_config = {
            'basis_gates': basis_gates,
            'config': config or {},
            'coupling_map': coupling_map,
            'seed': seed,
            'memory': memory
        }

        self._qjob_config = {'timeout': timeout} if self.is_local \
            else {'timeout': timeout, 'wait': wait}

        self._shared_circuits = False
        self._circuit_summary = False

        logger.info(self)

    def __str__(self):
        """Overload string.

        Retruns:
            str: the info of the object.
        """
        info = 'Qiskit Terra version {}\n'.format(terra_version)
        info += "Backend '{}', with following setting:\n{}\n{}\n{}\n{}".format(
            self.backend_name, self._backend_config, self._compile_config,
            self._run_config, self._qjob_config)
        return info

    def execute(self, circuits):
        """
        A wrapper to interface with quantum backend.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute

        Returns:
            Result: Result object
        """
        result = compile_and_run_circuits(circuits, self._backend, self._backend_config,
                                          self._compile_config, self._run_config,
                                          self._qjob_config,
                                          show_circuit_summary=self._circuit_summary,
                                          has_shared_circuits=self._shared_circuits)
        if self._circuit_summary:
            self._circuit_summary = False

        return result

    def set_config(self, **kwargs):
        for k, v in kwargs.items():
            if k in QuantumInstance.RUN_CONFIG:
                self._run_config[k] = v
            elif k in QuantumInstance.QJOB_CONFIG:
                self._qjob_config[k] = v
            elif k in QuantumInstance.COMPILE_CONFIG:
                self._compile_config[k] = v
            elif k in QuantumInstance.BACKEND_CONFIG:
                self._backend_config[k] = v
            elif k in QuantumInstance.SIMULATOR_CONFIG:
                self._backend_config['config'][k] = v
            else:
                raise ValueError("unknown setting for the key ({}).".format(k))

    @property
    def qjob_config(self):
        return self._qjob_config

    @property
    def backend_config(self):
        return self._backend_config

    @property
    def compile_config(self):
        return self._compile_config

    @property
    def run_config(self):
        return self._run_config

    @property
    def shared_circuits(self):
        return self._has_shared_circuits

    @shared_circuits.setter
    def shared_circuits(self, new_value):
        self._shared_circuits = new_value

    @property
    def circuit_summary(self):
        return self._circuit_summary

    @circuit_summary.setter
    def circuit_summary(self, new_value):
        self._circuit_summary = new_value

    @property
    def backend(self):
        """Return BaseBackend backend object."""
        return self._backend

    @property
    def backend_name(self):
        return self._backend.name()

    @property
    def is_statevector(self):
        """Return True if backend is a statevector-type simulator."""
        return QuantumInstance.is_statevector_backend(self._backend)

    @property
    def is_simulator(self):
        """Return True if backend is a simulator."""
        return QuantumInstance.is_simulator_backend(self._backend)

    @property
    def is_local(self):
        """Return True if backend is a local backend."""
        return QuantumInstance.is_local_backend(self._backend)

    @staticmethod
    def is_statevector_backend(backend):
        """
        Return True if backend object is statevector.

        Args:
            backend (BaseBackend): backend instance
        Returns:
            Result (Boolean): True is statevector
        """
        return backend.name().startswith('statevector') if backend is not None else False

    @staticmethod
    def is_simulator_backend(backend):
        """
        Returns True if backend is a simulator.

        Args:
            backend (BaseBackend): backend instance
        Returns:
            Result (Boolean): True is a simulator
        """
        return backend.configuration().simulator

    @staticmethod
    def is_local_backend(backend):
        """
        Returns True if backend is a local backend.

        Args:
            backend (BaseBackend): backend instance
        Returns:
            Result (Boolean): True is a local backend
        """
        return backend.configuration().local

    @staticmethod
    def register_and_get_operational_backends():
        # update registration info using internal methods because:
        # at this point I don't want to save to or remove credentials from disk
        # I want to update url, proxies etc without removing token and
        # re-adding in 2 methods

        ibmq_backends = []
        try:
            credentials = None
            preferences = Preferences()
            url = preferences.get_url()
            token = preferences.get_token()
            if url is not None and url != '' and token is not None and token != '':
                credentials = Credentials(token,
                                          url,
                                          proxies=preferences.get_proxies({}))
            if credentials is not None:
                IBMQ._accounts[credentials.unique_id()] = IBMQSingleProvider(
                    credentials, IBMQ)
                logger.debug("Registered with Qiskit successfully.")
                ibmq_backends = [x.name()
                                 for x in IBMQ.backends(url=url, token=token)]
        except Exception as e:
            logger.debug(
                "Failed to register with Qiskit: {}".format(str(e)))

        backends = set()
        aer_backends = [x.name() for x in get_aer_backends()]
        for backend in aer_backends:
            supported = True
            for unsupported_backend in QuantumInstance.UNSUPPORTED_BACKENDS:
                if backend.startswith(unsupported_backend):
                    supported = False
                    break

            if supported:
                backends.add(backend)

        return list(backends) + ibmq_backends


def get_quantum_instance_with_aer_statevector_simulator():
    backend = get_aer_backend('statevector_simulator')
    return QuantumInstance(backend)


def get_quantum_instance_with_aer_qasm_simulator(shots=1024):
    backend = get_aer_backend('qasm_simulator')
    return QuantumInstance(backend, shots=shots)
