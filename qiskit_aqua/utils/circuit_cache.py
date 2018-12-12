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
import uuid
from qiskit.backends.aer import AerJob
from qiskit.backends import JobError
from qiskit_aqua.aqua_error import AquaError
from qiskit import QuantumRegister
from qiskit.circuit import CompositeGate
import pickle
import logging
from qiskit.qobj import qobj_to_dict, Qobj

logger = logging.getLogger(__name__)

class CircuitCache:

    def __init__(self,
                 caching_naughty_mode = False,
                 cache_file = None):
        self.naughty_mode = caching_naughty_mode
        self.cache_file = cache_file
        self.misses = 0
        self.qobjs = []
        self.mappings = []
        self.try_reusing_qobjs = True

    def cache_circuit(self, qobj, circuits, chunk):
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

        self.qobjs.insert(chunk, copy.deepcopy(qobj))

        self.mappings.insert(chunk, [[] for i in range(len(circuits))])
        for circ_num, input_circuit in enumerate(circuits):
            # Delete qasm text, because it will be incorrect
            del self.qobjs[chunk].experiments[circ_num].header.compiled_circuit_qasm

            qreg_sizes = [reg.size for reg in input_circuit.qregs if isinstance(reg, QuantumRegister)]
            qreg_indeces = {reg.name: sum(qreg_sizes[0:i]) for i, reg in enumerate(input_circuit.qregs)}
            op_graph = {}

            # Unroll circuit in case of composite gates
            raw_gates = []
            for gate in input_circuit.data:
                if isinstance(gate, CompositeGate): raw_gates += gate.instruction_list()
                else: raw_gates += [gate]

            #TODO: See if we can skip gates with no params
            for i, uncompiled_gate in enumerate(raw_gates):
                regs = [(reg, qubit) for (reg, qubit) in uncompiled_gate.arg]
                qubits = [qubit+qreg_indeces[reg.name] for reg, qubit in regs if isinstance(reg, QuantumRegister)]
                gate_type = uncompiled_gate.name
                type_and_qubits = gate_type + qubits.__str__()
                op_graph[type_and_qubits] = \
                    op_graph.get(type_and_qubits, []) + [i]
            mapping = []
            for compiled_gate_index, compiled_gate in enumerate(qobj.experiments[circ_num].instructions):
                type_and_qubits = compiled_gate.name + compiled_gate.qubits.__str__()
                if len(op_graph[type_and_qubits]) > 0:
                    uncompiled_gate_index = op_graph[type_and_qubits].pop(0)
                    uncompiled_gate = raw_gates[uncompiled_gate_index]
                    regs = [(reg, qubit) for (reg, qubit) in uncompiled_gate.arg]
                    qubits = [qubit + qreg_indeces[reg.name] for reg, qubit in regs if isinstance(reg, QuantumRegister)]
                    if (compiled_gate.name == uncompiled_gate.name) and (compiled_gate.qubits.__str__() ==
                                                                         qubits.__str__()):
                        mapping.insert(compiled_gate_index, uncompiled_gate_index)
                else: raise Exception("Circuit shape does not match qobj, found extra {} instruction in qobj".format(
                    type_and_qubits))
            mappings[chunk][circ_num] = mapping
            for type_and_qubits, ops in op_graph.items():
                if len(ops) > 0:
                    raise Exception("Circuit shape does not match qobj, found extra {} in circuit".format(type_and_qubits))
        if self.cache_file is not None and len(self.cache_file) > 0:
            cache_handler = open(self.cache_file, 'wb')
            qobj_dicts = [qobj_to_dict(qob) for qob in self.qobjs]
            pickle.dump({'qobjs':qobj_dicts, 'mappings':self.mappings}, cache_handler, protocol=pickle.HIGHEST_PROTOCOL)
            cache_handler.close()
            logger.debug("Circuit cache saved to file.")

    def try_loading_cache_from_file(self):
        if len(self.qobjs) == 0 and self.cache_file is not None and len(self.cache_file) > 0:
            cache_handler = open(self.cache_file, "rb")
            cache = pickle.load(cache_handler, encoding="ASCII")
            cache_handler.close()
            self.qobjs = [Qobj.from_dict(qob) for qob in cache['qobjs']]
            self.mappings = cache['mappings']
            logger.debug("Circuit cache loaded from file.")

    # Note that this function overwrites the previous cached qobj for speed
    def load_qobj_from_cache(self, circuits, chunk):
        self.try_loading_cache_from_file()

        if self.try_reusing_qobjs and self.qobjs is not None and len(self.qobjs) <= chunk:
            self.mappings.insert(chunk, self.mappings[0])
            self.qobjs.insert(chunk, copy.deepcopy(qobjs[0]))

        for circ_num, input_circuit in enumerate(circuits):

            # If there are too few experiments in the cache, try reusing the first experiment.
            # Only do this for the first chunk. Subsequent chunks should rely on these copies through the deepcopy above.
            if self.try_reusing_qobjs and chunk == 0 and circ_num > 0 and len(self.qobjs[chunk].experiments) <= \
                    circ_num:
                self.qobjs[0].experiments.insert(circ_num, copy.deepcopy(self.qobjs[0].experiments[0]))
                self.mappings[0].insert(circ_num, self.mappings[0][0])

            # Unroll circuit in case of composite gates
            raw_gates = []
            for gate in input_circuit.data:
                if isinstance(gate, CompositeGate): raw_gates += gate.instruction_list()
                else: raw_gates += [gate]
            self.qobjs[chunk].experiments[circ_num].header.name = input_circuit.name
            for gate_num, compiled_gate in enumerate(self.qobjs[chunk].experiments[circ_num].instructions):
                if compiled_gate.name == 'snapshot': continue
                cache_index = self.mappings[chunk][circ_num][gate_num]
                uncompiled_gate = raw_gates[cache_index]

                # Need the 'getattr' wrapper because measure has no 'params' field and breaks this.
                if not len(getattr(compiled_gate, 'params', [])) == len(getattr(uncompiled_gate, 'param', [])) or \
                    not compiled_gate.name == uncompiled_gate.name:
                    raise AquaError('Gate mismatch at gate {0} ({1}, {2} params) of circuit against '
                                         'gate {3} ({4}, {5} params) '
                                         'of cached qobj'.format(cache_index, uncompiled_gate.name, len(uncompiled_gate.param),
                                                                 gate_num, compiled_gate.name, len(compiled_gate.params)))
                compiled_gate.params = np.array(uncompiled_gate.param, dtype=float).tolist()
        exec_qobj = copy.copy(self.qobjs[chunk])
        if self.naughty_mode: exec_qobj.experiments = self.qobjs[chunk].experiments[0:len(circuits)]
        else: exec_qobj.experiments = copy.deepcopy(self.qobjs[chunk].experiments[0:len(circuits)])
        return exec_qobj

    # Does what backend.run and aerjob.submit do, but without qobj validation.
    def naughty_run(self, backend, qobj):
        job_id = str(uuid.uuid4())
        aer_job = AerJob(backend, job_id, backend._run_job, qobj)
        if aer_job._future is not None:
            raise JobError("We have already submitted the job!")
        aer_job._future = aer_job._executor.submit(aer_job._fn, aer_job._job_id, aer_job._qobj)
        return aer_job

    def clear_cache(self):
        self.qobjs = []
        self.mappings = []
        self.misses = 0
        self.try_reusing_qobjs = True