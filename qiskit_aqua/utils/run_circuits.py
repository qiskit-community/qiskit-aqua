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

import numpy as np
import qiskit
from qiskit import compile as q_compile
from qiskit.backends.jobstatus import JobStatus
from qiskit.backends import JobError

from qiskit_aqua.algorithmerror import AlgorithmError
from qiskit_aqua.utils import summarize_circuits

logger = logging.getLogger(__name__)


def run_circuits(circuits, backend, execute_config, qjob_config={},
                 max_circuits_per_job=sys.maxsize, show_circuit_summary=False):
    """
    An execution wrapper with Qiskit-Terra, with job auto recover capability.

    The autorecovery feature is only applied for non-simulator backend.
    This wraper will try to get the result no matter how long it costs.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (str): name of backend
        execute_config (dict): settings for qiskit execute (or compile)
        qjob_config (dict): settings for job object, like timeout and wait
        max_circuits_per_job (int): the maximum number of job, default is unlimited but 300
                is limited if you submit to a remote backend
        show_circuit_summary (bool): showing the summary of submitted circuits.

    Returns:
        Result: Result object

    Raises:
        AlgorithmError: Any error except for JobError raised by Qiskit Terra
    """

    if not isinstance(circuits, list):
        circuits = [circuits]

    my_backend = None
    try:
        my_backend = qiskit.Aer.get_backend(backend)
    except KeyError:
        my_backend = qiskit.IBMQ.get_backend(backend)

    with_autorecover = False if my_backend.configuration()[
        'simulator'] else True

    support_qobj = my_backend.configuration().get('allow_q_object', False)
    job_completed_signature = 'COMPLETED' if not support_qobj else 'Successful completion'

    qobjs = []
    jobs = []
    chunks = int(np.ceil(len(circuits) / max_circuits_per_job))

    for i in range(chunks):
        sub_circuits = circuits[i *
                                max_circuits_per_job:(i + 1) * max_circuits_per_job]
        qobj = q_compile(sub_circuits, my_backend, **execute_config)
        job = my_backend.run(qobj)
        jobs.append(job)
        qobjs.append(qobj)

    if logger.isEnabledFor(logging.DEBUG) and show_circuit_summary:
        logger.debug(summarize_circuits(circuits))

    results = []
    if with_autorecover:

        logger.info("There are {} circuits and they are chunked into "
                    "{} chunks, each with {} circutis.".format(len(circuits), chunks, max_circuits_per_job))

        for idx in range(len(jobs)):
            job = jobs[idx]
            job_id = job.id()
            logger.info(
                "Running {}-th chunk circuits, job id: {}".format(idx, job_id))
            while True:
                try:
                    result = job.result(**qjob_config)
                    if result.status == job_completed_signature:
                        results.append(result)
                        logger.info(
                            "COMPLETED the {}-th chunk of circuits, job id: {}".format(idx, job_id))
                        break
                    else:
                        logger.warning(
                            "FAILURE: the {}-th chunk of circuits, job id: {}".format(idx, job_id))
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
                                       "status: 'FAIL_TO_GET_STATUS' Terra job error: {}".format(job_id, e))
                        time.sleep(5)
                    except Exception as e:
                        raise AlgorithmError("FAILURE: job id: {}, "
                                             "status: 'FAIL_TO_GET_STATUS' ({})".format(job_id, e)) from e

                logger.info("Job status: {}".format(job_status))
                # when reach here, it means the job fails. let's check what kinds of failure it is.
                if job_status == JobStatus.DONE:
                    logger.info(
                        "Job ({}) is completed anyway, retrieve result from backend.".format(job_id))
                    job = my_backend.retrieve_job(job_id)
                elif job_status == JobStatus.RUNNING or job_status == JobStatus.QUEUED:
                    logger.info("Job ({}) is {}, but encounter an exception, "
                                "recover it from backend.".format(job_id, job_status))
                    job = my_backend.retrieve_job(job_id)
                else:
                    logger.info(
                        "Fail to run Job ({}), resubmit it.".format(job_id))
                    qobj = qobjs[idx]
                    job = my_backend.run(qobj)
    else:
        results = []
        for job in jobs:
            results.append(job.result(**qjob_config))

    if len(results) != 0:
        result = functools.reduce(lambda x, y: x + y, results)
    else:
        result = None
    return result
