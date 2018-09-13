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
from qiskit import register as q_register
from qiskit import unregister as q_unregister
from qiskit import registered_providers as q_registered_providers
from qiskit import execute as q_execute, compile as q_compile
from qiskit import available_backends, get_backend
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.jobstatus import JobStatus

from qiskit_aqua import AlgorithmError
from qiskit_aqua.utils import summarize_circuits
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
        'local_unitary_simulator', 'local_clifford_simulator']

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

    def setup_quantum_backend(self, backend='local_statevector_simulator', shots=1024, skip_transpiler=False,
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
        if self.EQUIVALENT_BACKENDS.get(backend, backend) not in operational_backends:
            raise AlgorithmError("This backend '{}' is not operational for the quantum algorithm, \
                                 select any one below: {}".format(backend, operational_backends))

        self._backend = backend
        self._qjob_config = {'timeout': timeout,
                             'wait': wait}

        shots = 1 if 'statevector' in backend else shots
        noise_params = noise_params if 'simulator' in backend else None

        if backend.startswith('local'):
            self._qjob_config.pop('wait', None)
            self.MAX_CIRCUITS_PER_JOB = sys.maxsize

        my_backend = get_backend(backend)

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
        result = self.execute_with_autorecover(circuits, self._backend, self._execute_config,
                                               self._qjob_config, max_circuits_per_job=self.MAX_CIRCUITS_PER_JOB,
                                               show_circuit_summary=self._show_circuit_summary)
        if self._show_circuit_summary:
            self.disable_circuit_summary()

        return result

    @staticmethod
    def execute_with_maybe_autorecover(circuits, backend, execute_config, qjob_config={},
                                       max_circuits_per_job=sys.maxsize, show_circuit_summary=False):
        """
        An execution wrapper with qiskit-terra, with auto recover capability.

        By default, simulator backend will not use autorecover feature.

        Args:
            circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
            backend (str): name of backend
            execute_config (dict): settings for qiskit execute (or compile)
            qjob_config (dict): settings for job object, like timeout and wait
            max_circuits_per_job (int): the maximum number of job, default is unlimited but 300 is limited if you
                                        submit to a remote backend
            show_circuit_summary (bool): showing the summary of submitted circuits.

        Returns:
            Result: Result object
        """

        # max_circuits_per_job = QuantumAlgorithm.MAX_CIRCUITS_PER_JOB
        if not isinstance(circuits, list):
            circuits = [circuits]

        with_autorecover = False if my_backend.configuration()['simulator'] else True

        my_backend = get_backend(backend)
        qobjs = []
        jobs = []
        chunks = int(np.ceil(len(circuits) / max_circuits_per_job))

        for i in range(chunks):
            sub_circuits = circuits[i * max_circuits_per_job:(i + 1) * max_circuits_per_job]
            qobj = q_compile(sub_circuits, my_backend, **execute_config)
            job = my_backend.run(qobj)
            jobs.append(job)
            qobjs.append(qobj)

        logger.info("There are {} circuits and they are chunked into {} chunks, each with {} circutis.".format(
            len(circuits), chunks, max_circuits_per_job))

        if logger.isEnabledFor(logging.DEBUG) and show_circuit_summary:
            logger.debug(summarize_circuits(circuits))

        results = []

        if with_autorecover:
            for idx in range(len(jobs)):
                job = jobs[idx]
                logger.info("Running {}-th chunk circuits, job id: {}".format(idx, job.id()))
                while True:
                    try:
                        result = job.result(**qjob_config)
                        if result.status == 'COMPLETED':
                            results.append(result)
                            logger.info("COMPLETED the {}-th chunk of circuits, job id: {}".format(idx, job.id()))
                            break
                        else:
                            logger.warning(
                                "FAILURE: the {}-th chunk of circuits, job id: {}, status: {}".format(idx, job.id(), job.status()))
                    except Exception as e:
                        # if terra raise any error, which means something wrong, re-run it
                        logger.warning(
                            "FAILURE: the {}-th chunk of circuits, job id: {}, status: {}, Terra error: {} ".format(idx, job.id(), job.status(), e))
                    # when reach here, it means the job fails. let's check what kinds of failure it is.
                    if job.status() == JobStatus.DONE:
                        logger.info("Job ({}) is completed anyway, retrieve result from backend.".format(job.id()))
                        job = my_backend.retrieve_job(job.id())
                    elif job.status() == JobStatus.RUNNING or job.status() == JobStatus.QUEUED:
                        logger.info("Job ({}) is {}, but encounter an exception, recover it from backend.".format(
                            job.id(), job.status()))
                        job = my_backend.retrieve_job(job.id())
                    else:
                        logger.info("Fail to run Job ({}), resubmit it.".format(job.id()))
                        qobj = qobjs[idx]
                        job = my_backend.run(qobj)
        else:
            results = []
            for job in jobs:
                results.append(job.result(**qjob_config))

        if len(results) != 0:
            result = functools.reduce(lambda x, y: x + y, results)

        return result

    @staticmethod
    def register_and_get_operational_backends(*args, provider_class=IBMQProvider, **kwargs):
        try:
            for provider in q_registered_providers():
                if isinstance(provider, provider_class):
                    q_unregister(provider)
                    logger.debug(
                        "Provider '{}' unregistered with Qiskit successfully.".format(provider_class))
                    break
        except Exception as e:
            logger.debug(
                "Failed to unregister provider '{}' with Qiskit: {}".format(provider_class, str(e)))

        preferences = Preferences()
        if args or kwargs or preferences.get_token() is not None:
            try:
                q_register(*args, provider_class=provider_class, **kwargs)
                logger.debug(
                    "Provider '{}' registered with Qiskit successfully.".format(provider_class))
            except Exception as e:
                logger.debug(
                    "Failed to register provider '{}' with Qiskit: {}".format(provider_class, str(e)))

        backends = available_backends()
        backends = [
            x for x in backends if x not in QuantumAlgorithm.UNSUPPORTED_BACKENDS]
        return backends

    @abstractmethod
    def init_params(self, params, algo_input):
        pass

    @abstractmethod
    def run(self):
        pass
