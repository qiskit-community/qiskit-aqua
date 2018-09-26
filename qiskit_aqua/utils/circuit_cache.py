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

""" A utility for caching and reparameterizing circuits, rather than compiling from scratch
with each iteration. This is a singleton style module, with only a single instance per pytbon
runtime. Each time an algorithm begins, the cache is wiped. Note that caching only works when
transpilation is off (aqua_dict['backend']['skip_transpiler'] = True).

Caching is controlled via the aqua_dict['problem']['circuit_caching'] parameter. Caching naughty
mode bypasses qobj validation before compiling and reuses the same qobj object over and over to
avoid deepcopying. It is controlled via the aqua_dict['problem']['caching_naughty_mode'] parameter.
Note that naughty mode only works for local simulation.

You may also specify a filename into which to store the cache as a pickle file, for molecules which
are expensive to compile even the first time. The filename is set in aqua_dict['problem']['circuit_cache_file'].
If a filename is present, the system will attempt to load from the file. In the event of an error, the system
will fail gracefully, compile from scratch, and cache the new compiled qobj and mapping in the file location in pickled
form."""

import numpy as np
import copy
from qiskit.backends.local import LocalJob
from qiskit.backends import JobError
import pickle
import logging
from qiskit.qobj import qobj_to_dict, Qobj

logger = logging.getLogger(__name__)

qobjs = None
mappings = None
misses = 0
use_caching = False
naughty_mode = False
cache_file = None

def cache_circuit(qobj, circuits, chunk):
    """
    A helper method for caching compiled qobjs by storing the compiled qobj
    and constructing a mapping array from the uncompiled operations in the circuit
    to the instructions in the qobj. Note that the "qobjs" list in the cache dict is a
    list of the cached chunks, each element of which contains a single qobj with as
    many experiments as is allowed by the execution backend. E.g. if the backend allows
    300 experiments per job and the user wants to run 500 circuits,
    len(circuit_cache['qobjs']) == 2,
    len(circuit_cache['qobjs'][0].experiments) == 300, and
    len(circuit_cache['qobjs'][1].experiments) == 200.

    This feature is only applied if 'circuit_caching' is True in the 'problem' Aqua
    dictionary section and 'skip_transpiler' is True in the 'backend' section. Note that
    the global circuit_cache is defined inside algomethods.py.

    Args:
        #TODO
    """
    global qobjs
    global mappings
    global misses
    global cache_file

    if qobjs is None: qobjs = []
    if mappings is None: mappings = []
    if misses is None: misses = 0

    qobjs.insert(chunk, copy.deepcopy(qobj))

    mappings.insert(chunk, [[] for i in range(len(circuits))])
    for circ_num, input_circuit in enumerate(circuits):
        # Delete qasm text, because it will be incorrect
        del qobjs[chunk].experiments[circ_num].header.compiled_circuit_qasm
        op_graph = {}
        for i, uncompiled_gate in enumerate(input_circuit.data):
            if uncompiled_gate.name == 'measure' : qubits = [uncompiled_gate.arg[0][1]]
            else: qubits = uncompiled_gate._qubit_coupling
            gate_type = uncompiled_gate.name
            type_and_qubits = gate_type + qubits.__str__()
            op_graph[type_and_qubits] = \
                op_graph.get(type_and_qubits, []) + [i]
        mapping = []
        for compiled_gate_index, compiled_gate in enumerate(qobj.experiments[circ_num].instructions):
            type_and_qubits = compiled_gate.name + compiled_gate.qubits.__str__()
            if len(op_graph[type_and_qubits]) > 0:
                uncompiled_gate_index = op_graph[type_and_qubits].pop(0)
                uncompiled_gate = input_circuit.data[uncompiled_gate_index]
                if uncompiled_gate.name == 'measure': qubits = [uncompiled_gate.arg[0][1]]
                else: qubits = uncompiled_gate._qubit_coupling
                if (compiled_gate.name == uncompiled_gate.name) and (compiled_gate.qubits.__str__() ==
                                                                     qubits.__str__()):
                    mapping.insert(compiled_gate_index, uncompiled_gate_index)
            else: raise Exception("Circuit shape does not match qobj, found extra {} instruction in qobj".format(
                type_and_qubits))
        mappings[chunk][circ_num] = mapping
        for type_and_qubits, ops in op_graph.items():
            if len(ops) > 0:
                raise Exception("Circuit shape does not match qobj, found extra {} in circuit".format(type_and_qubits))
    if len(cache_file) > 0:
        cache_handler = open(cache_file, 'wb')
        # qobj_jsons = [json.dumps(qobj_to_dict(qob)) for qob in qobjs]
        qobj_dicts = [qobj_to_dict(qob) for qob in qobjs]
        pickle.dump({'qobjs':qobj_dicts, 'mappings':mappings}, cache_handler, protocol=pickle.HIGHEST_PROTOCOL)
        cache_handler.close()
        logger.debug("Circuit cache saved successfully.")

# Note that this function overwrites the previous cached qobj for speed
def load_qobj_from_cache(circuits, chunk):
    global qobjs
    global mappings
    global misses
    global cache_file

    if qobjs is None and len(cache_file) > 0:
        cache_handler = open(cache_file, "rb")
        cache = pickle.load(cache_handler, encoding="ASCII")
        cache_handler.close()
        qobjs = [Qobj.from_dict(qob) for qob in cache['qobjs']]
        mappings = cache['mappings']
        logger.debug("Circuit cache loaded successfully.")

    for circ_num, input_circuit in enumerate(circuits):
        qobjs[chunk].experiments[circ_num].header.name = input_circuit.name
        for gate_num, compiled_gate in enumerate(qobjs[chunk].experiments[circ_num].instructions):
            if compiled_gate.name == 'snapshot': continue
            uncompiled_gate = input_circuit.data[mappings[chunk][circ_num][gate_num]]
            compiled_gate.params = np.array(uncompiled_gate.param, dtype=float).tolist()
    if naughty_mode: return qobjs[chunk]
    else: return copy.deepcopy(qobjs[chunk])

def naughty_run(backend, qobj):
    local_job = LocalJob(backend._run_job, qobj)
    if local_job._future is not None:
        raise JobError("We have already submitted the job!")
    local_job._future = local_job._executor.submit(local_job._fn, local_job._qobj)
    return local_job

def clear_cache():
    global qobjs
    global mappings
    global misses

    qobjs = None
    mappings = None
    misses = 0