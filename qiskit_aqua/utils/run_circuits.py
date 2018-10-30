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

import sys
import logging
import time
import functools
import copy
from packaging import version

import numpy as np
from qiskit.backends import BaseBackend
from qiskit import compile as q_compile
import qiskit
from qiskit.backends.jobstatus import JobStatus
from qiskit.backends import JobError

from qiskit_aqua.algorithmerror import AlgorithmError
from qiskit_aqua.utils import summarize_circuits

logger = logging.getLogger(__name__)

MAX_CIRCUITS_PER_JOB = 300


def _avoid_empty_circuits(circuits):

    new_circuits = []
    for qc in circuits:
        if len(qc) == 0:
            tmp_q = None
            for q_name, q in qc.get_qregs().items():
                tmp_q = q
                break
            if tmp_q is None:
                raise AlgorithmError("A QASM without any quantum register is invalid.")
            qc.iden(tmp_q[0])
        new_circuits.append(qc)
    return new_circuits


def _reuse_shared_circuits(circuits, backend, execute_config, qjob_config={}):
    """Reuse the circuits with the shared head.

    We assume the 0-th circuit is the shared_circuit, so we execute it first
    and then use it as initial state for simulation.

    Note that all circuits should have the exact the same shared parts.

    TODO:
        after subtraction, the circuits can not be empty.
        it only works for terra 0.6.2+
    """
    shared_circuit = circuits[0]
    shared_result = run_circuits(shared_circuit, backend, execute_config,
                                 show_circuit_summary=True)

    if len(circuits) == 1:
        return shared_result
    shared_quantum_state = np.asarray(shared_result.get_statevector(shared_circuit))
    # extract different of circuits
    for circuit in circuits[1:]:
        circuit.data = circuit.data[len(shared_circuit):]

    temp_execute_config = copy.deepcopy(execute_config)
    if 'config' not in temp_execute_config:
        temp_execute_config['config'] = dict()
    temp_execute_config['config']['initial_state'] = shared_quantum_state
    diff_result = run_circuits(circuits[1:], backend, temp_execute_config, qjob_config)
    result = shared_result + diff_result
    return result

def run_circuits(circuits, backend, execute_config, qjob_config={},
                 show_circuit_summary=False, has_shared_circuits=False):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): backend instance
        execute_config (dict): settings for qiskit execute (or compile)
        qjob_config (dict): settings for job object, like timeout and wait
        show_circuit_summary (bool): showing the summary of submitted circuits.
        has_shared_circuits (bool): use the 0-th circuits as initial state for other circuits.
    Returns:
        Result: Result object

    Raises:
        AlgorithmError: Any error except for JobError raised by Qiskit Terra
    """
    if backend is None or not isinstance(backend, BaseBackend):
        raise AlgorithmError('Backend is missing or not an instance of BaseBackend')

    if not isinstance(circuits, list):
        circuits = [circuits]

    if backend.configuration().get('name', '').startswith('statevector'):
        circuits = _avoid_empty_circuits(circuits)

    if has_shared_circuits and version.parse(qiskit.__version__) > version.parse('0.6.1'):
        return _reuse_shared_circuits(circuits, backend, execute_config, qjob_config)

    with_autorecover = False if backend.configuration()['simulator'] else True
    max_circuits_per_job = sys.maxsize if backend.configuration()['local'] \
        else MAX_CIRCUITS_PER_JOB

    qobjs = []
    jobs = []
    chunks = int(np.ceil(len(circuits) / max_circuits_per_job))

    for i in range(chunks):
        sub_circuits = circuits[i *
                                max_circuits_per_job:(i + 1) * max_circuits_per_job]
        qobj = q_compile(sub_circuits, backend, **execute_config)
        job = backend.run(qobj)
        jobs.append(job)
        qobjs.append(qobj)

    if logger.isEnabledFor(logging.DEBUG) and show_circuit_summary:
        logger.debug(summarize_circuits(circuits))

    results = []
    if with_autorecover:

        logger.debug("There are {} circuits and they are chunked into {} chunks, "
                     "each with {} circutis.".format(len(circuits), chunks,
                                                     max_circuits_per_job))

        for idx in range(len(jobs)):
            job = jobs[idx]
            job_id = job.job_id()
            logger.info("Running {}-th chunk circuits, job id: {}".format(idx, job_id))
            while True:
                try:
                    result = job.result(**qjob_config)
                    if result.status == 'COMPLETED':
                        results.append(result)
                        logger.info("COMPLETED the {}-th chunk of circuits, "
                                    "job id: {}".format(idx, job_id))
                        break
                    else:
                        logger.warning("FAILURE: the {}-th chunk of circuits, "
                                       "job id: {}".format(idx, job_id))
                except JobError as e:
                    # if terra raise any error, which means something wrong, re-run it
                    logger.warning("FAILURE: the {}-th chunk of circuits, job id: {}, "
                                   "Terra job error: {} ".format(idx, job_id, e))
                except Exception as e:
                    raise AlgorithmError("FAILURE: the {}-th chunk of circuits, job id: {}, "
                                         "Terra unknown error: {} ".format(idx, job_id, e)) from e

                # keep querying the status until it is okay.
                while True:
                    try:
                        job_status = job.status()
                        break
                    except JobError as e:
                        logger.warning("FAILURE: job id: {}, "
                                       "status: 'FAIL_TO_GET_STATUS' "
                                       "Terra job error: {}".format(job_id, e))
                        time.sleep(5)
                    except Exception as e:
                        raise AlgorithmError("FAILURE: job id: {}, "
                                             "status: 'FAIL_TO_GET_STATUS' "
                                             "({})".format(job_id, e)) from e

                logger.info("Job status: {}".format(job_status))
                # when reach here, it means the job fails. let's check what kinds of failure it is.
                if job_status == JobStatus.DONE:
                    logger.info("Job ({}) is completed anyway, retrieve result "
                                "from backend.".format(job_id))
                    job = backend.retrieve_job(job_id)
                elif job_status == JobStatus.RUNNING or job_status == JobStatus.QUEUED:
                    logger.info("Job ({}) is {}, but encounter an exception, "
                                "recover it from backend.".format(job_id, job_status))
                    job = backend.retrieve_job(job_id)
                else:
                    logger.info("Fail to run Job ({}), resubmit it.".format(job_id))
                    qobj = qobjs[idx]
                    job = backend.run(qobj)
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    if len(results) != 0:
        result = functools.reduce(lambda x, y: x + y, results)
    else:
        result = None
    return result
