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

qobj_cache = {'qobjs':[], 'mappings':[]}

def cache_qobj(qobj, circuits, chunk):
    """
    A helper method for caching compiled qobjs by storing the compiled qobj
    and constructing a mapping array from the uncompiled operations in the circuit
    to the instructions in the qobj. Note that the "qobjs" list in the cache dict is a
    list of the cached chunks, each element of which contains a single qobj with as
    many experiments as is allowed by the execution backend. E.g. if the backend allows
    300 experiments per job and the user wants to run 500 circuits,
    len(qobj_cache['qobjs']) == 2,
    len(qobj_cache['qobjs'][0].experiments) == 300, and
    len(qobj_cache['qobjs'][1].experiments) == 200.

    This feature is only applied if 'use_qobj_caching' is True in the 'problem' Aqua
    dictionary section and 'skip_transpiler' is True in the 'backend' section.

    Args:
        #TODO
    """
    qobj_cache['mappings'].insert(chunk, [[] for i in range(len(circuits))])
    for circ_num, input_circuit in enumerate(circuits):
        op_graph = {}
        for i, uncompiled_gate in enumerate(input_circuit.data):
            qubits = uncompiled_gate._qubit_coupling
            gate_type = uncompiled_gate.name
            type_and_qubits = gate_type + qubits.__str__()
            op_graph[type_and_qubits] = \
                op_graph.get(type_and_qubits, []) + [i]
        mapping = []
        for i, compiled_gate in enumerate(qobj.experiments[circ_num].instructions):
            if compiled_gate.name == 'snapshot': continue
            type_and_qubits = compiled_gate.name + compiled_gate.qubits.__str__()
            if len(op_graph[type_and_qubits]) > 0:
                uncompiled_gate_index = op_graph[type_and_qubits].pop(0)
                uncompiled_gate = input_circuit.data[uncompiled_gate_index]
                if (compiled_gate.name == uncompiled_gate.name) and (compiled_gate.qubits.__str__() ==
                                                                     uncompiled_gate._qubit_coupling.__str__()):
                    mapping.insert(i, uncompiled_gate_index)
            else: raise Exception("Circuit shape does not match qobj, found extra {} instruction in qobj".format(
                type_and_qubits))
        # if circ_num not in qobj_cache['mappings'][chunk][circ_num]
        qobj_cache['mappings'][chunk].append(mapping)
        qobj_cache['qobjs'].insert(chunk, qobj)
        for type_and_qubits, ops in op_graph.items():
            if len(ops) > 0:
                raise Exception("Circuit shape does not match qobj, found extra {} in circuit".format(type_and_qubits))
        # check if op_graph is empty to confirm correct circuit shape

# Note that this function overwrites the previous cached qobj for speed
def load_qobj_from_cache(circuits, cached_qobj_chunk):
    for i, input_circuit in enumerate(circuits):
        cached_qobj_chunk.experiments[i].header.name = input_circuit.name
        for i, compiled_gate in enumerate(cached_qobj_chunk.experiments[i].instructions):
            if compiled_gate.name == 'snapshot': continue
            uncompiled_gate = input_circuit.data[self.mapping[i]]
            compiled_gate.params = np.array(uncompiled_gate.param, dtype=float).tolist()
    return cached_qobj_chunk

def run_circuits(circuits, backend, execute_config, qjob_config={},
                 max_circuits_per_job=sys.maxsize, show_circuit_summary=False, use_qobj_caching=True):
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

    qobjs = []
    jobs = []
    chunks = int(np.ceil(len(circuits) / max_circuits_per_job))

    for i in range(chunks):
<<<<<<< HEAD
        sub_circuits = circuits[i *
                                max_circuits_per_job:(i + 1) * max_circuits_per_job]
        qobj = q_compile(sub_circuits, my_backend, **execute_config)
=======
        sub_circuits = circuits[i * max_circuits_per_job:(i + 1) * max_circuits_per_job]
        if i not in qobj_cache['qobjs'] or not use_qobj_caching:
            qobj = q_compile(sub_circuits, my_backend, **execute_config)
            if use_qobj_caching: cache_qobj(qobj, circuits, i)
        elif use_qobj_caching:
            qobj = load_qobj_from_cache(sub_circuits, qobj_cache['qobjs'][i])
>>>>>>> Working caching and loading from cache
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
                    if result.status == 'COMPLETED':
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
