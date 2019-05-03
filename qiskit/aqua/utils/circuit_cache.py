# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" A utility for caching and reparameterizing circuits, rather than compiling from scratch
with each iteration. Note that if the circuit is transpiled aggressively such that rotation parameters
cannot be easily mapped from the uncompiled to compiled circuit, caching will fail gracefully to
standard compilation. This will be noted by multiple cache misses in the DEBUG log. It is generally safer to
skip the transpiler (aqua_dict['backend']['skip_transpiler'] = True) when using caching.

Caching is controlled via the aqua_dict['problem']['circuit_caching'] parameter. Setting skip_qobj_deepcopy = True
reuses the same qobj object over and over to avoid deepcopying. It is controlled via the aqua_dict['problem'][
'skip_qobj_deepcopy'] parameter.

You may also specify a filename into which to store the cache as a pickle file, for circuits which
are expensive to compile even the first time. The filename is set in aqua_dict['problem']['circuit_cache_file'].
If a filename is present, the system will attempt to load from the file.

In the event of an error, the system will fail gracefully, compile from scratch, and cache the new
compiled qobj and mapping in the file location in pickled form. It will fail over 5 times before deciding
that caching should be disabled."""

import numpy as np
import copy
import pickle
import logging

from qiskit import QuantumRegister
from qiskit.circuit import CompositeGate
from qiskit.assembler.run_config import RunConfig
from qiskit.qobj import Qobj, QasmQobjConfig

from qiskit.aqua.aqua_error import AquaError

logger = logging.getLogger(__name__)


class CircuitCache:

    def __init__(self,
                 skip_qobj_deepcopy=False,
                 cache_file=None,
                 allowed_misses=3):
        self.skip_qobj_deepcopy = skip_qobj_deepcopy
        self.cache_file = cache_file
        self.misses = 0
        self.qobjs = []
        self.mappings = []
        self.cache_transpiled_circuits = False
        self.try_reusing_qobjs = True
        self.allowed_misses = allowed_misses
        try:
            self.try_loading_cache_from_file()
        except(EOFError, FileNotFoundError) as e:
            logger.warning("Error loading cache from file {0}: {1}".format(self.cache_file, repr(e)))

    def cache_circuit(self, qobj, circuits, chunk):
        """
        A method for caching compiled qobjs by storing the compiled qobj
        and constructing a mapping array from the uncompiled operations in the circuit
        to the instructions in the qobj. Note that the "qobjs" list in the cache dict is a
        list of the cached chunks, each element of which contains a single qobj with as
        many experiments as is allowed by the execution backend. E.g. if the backend allows
        300 experiments per job and the user wants to run 500 circuits,
        len(circuit_cache['qobjs']) == 2,
        len(circuit_cache['qobjs'][0].experiments) == 300, and
        len(circuit_cache['qobjs'][1].experiments) == 200.

        This feature is only applied if 'circuit_caching' is True in the 'problem' Aqua
        dictionary section.

        Args:
            qobj (Qobj): A compiled qobj to be saved
            circuits (list): The original uncompiled QuantumCircuits
            chunk (int): If a larger list of circuits was broken into chunks by run_algorithm for separate runs,
            which chunk number `circuits` represents
        """

        self.qobjs.insert(chunk, copy.deepcopy(qobj))

        self.mappings.insert(chunk, [{} for i in range(len(circuits))])
        for circ_num, input_circuit in enumerate(circuits):

            qreg_sizes = [reg.size for reg in input_circuit.qregs if isinstance(reg, QuantumRegister)]
            qreg_indeces = {reg.name: sum(qreg_sizes[0:i]) for i, reg in enumerate(input_circuit.qregs)}
            op_graph = {}

            # Unroll circuit in case of composite gates
            raw_gates = []
            for gate in input_circuit.data:
                if isinstance(gate, CompositeGate): raw_gates += gate.instruction_list()
                else: raw_gates += [gate]

            for i, (uncompiled_gate, regs, _) in enumerate(raw_gates):
                if not hasattr(uncompiled_gate, 'params') or len(uncompiled_gate.params) < 1: continue
                if uncompiled_gate.name == 'snapshot': continue
                qubits = [qubit+qreg_indeces[reg.name] for reg, qubit in regs if isinstance(reg, QuantumRegister)]
                gate_type = uncompiled_gate.name
                type_and_qubits = gate_type + qubits.__str__()
                op_graph[type_and_qubits] = \
                    op_graph.get(type_and_qubits, []) + [i]
            mapping = {}
            for compiled_gate_index, compiled_gate in enumerate(qobj.experiments[circ_num].instructions):
                if not hasattr(compiled_gate, 'params') or len(compiled_gate.params) < 1: continue
                if compiled_gate.name == 'snapshot': continue
                type_and_qubits = compiled_gate.name + compiled_gate.qubits.__str__()
                if len(op_graph[type_and_qubits]) > 0:
                    uncompiled_gate_index = op_graph[type_and_qubits].pop(0)
                    (uncompiled_gate, regs, _) = raw_gates[uncompiled_gate_index]
                    qubits = [qubit + qreg_indeces[reg.name] for reg, qubit in regs if isinstance(reg, QuantumRegister)]
                    if (compiled_gate.name == uncompiled_gate.name) and (compiled_gate.qubits.__str__() ==
                                                                         qubits.__str__()):
                        mapping[compiled_gate_index] = uncompiled_gate_index
                else: raise AquaError("Circuit shape does not match qobj, found extra {} instruction in qobj".format(
                    type_and_qubits))
            self.mappings[chunk][circ_num] = mapping
            for type_and_qubits, ops in op_graph.items():
                if len(ops) > 0:
                    raise AquaError("Circuit shape does not match qobj, found extra {} in circuit".format(type_and_qubits))
        if self.cache_file is not None and len(self.cache_file) > 0:
            with open(self.cache_file, 'wb') as cache_handler:
                qobj_dicts = [qob.to_dict() for qob in self.qobjs]
                pickle.dump({'qobjs': qobj_dicts,
                             'mappings': self.mappings,
                             'transpile': self.cache_transpiled_circuits},
                            cache_handler,
                            protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug("Circuit cache saved to file: {}".format(self.cache_file))

    def try_loading_cache_from_file(self):
        if len(self.qobjs) == 0 and self.cache_file is not None and len(self.cache_file) > 0:
            with open(self.cache_file, "rb") as cache_handler:
                try:
                    cache = pickle.load(cache_handler, encoding="ASCII")
                except (EOFError) as e:
                    logger.debug("No cache found in file: {}".format(self.cache_file))
                    return
                self.qobjs = [Qobj.from_dict(qob) for qob in cache['qobjs']]
                self.mappings = cache['mappings']
                self.cache_transpiled_circuits = cache['transpile']
                logger.debug("Circuit cache loaded from file: {}".format(self.cache_file))

    # Note that this function overwrites the previous cached qobj for speed
    def load_qobj_from_cache(self, circuits, chunk, run_config=None):
        self.try_loading_cache_from_file()

        if self.try_reusing_qobjs and self.qobjs is not None and len(self.qobjs) > 0 and len(self.qobjs) <= chunk:
            self.mappings.insert(chunk, self.mappings[0])
            self.qobjs.insert(chunk, copy.deepcopy(self.qobjs[0]))

        for circ_num, input_circuit in enumerate(circuits):

            # If there are too few experiments in the cache, try reusing the first experiment.
            # Only do this for the first chunk. Subsequent chunks should rely on these copies
            # through the deepcopy above.
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
                if not hasattr(compiled_gate, 'params') or len(compiled_gate.params) < 1: continue
                if compiled_gate.name == 'snapshot': continue
                cache_index = self.mappings[chunk][circ_num][gate_num]
                (uncompiled_gate, regs, _) = raw_gates[cache_index]

                # Need the 'getattr' wrapper because measure has no 'params' field and breaks this.
                if not len(getattr(compiled_gate, 'params', [])) == len(getattr(uncompiled_gate, 'params', [])) or \
                    not compiled_gate.name == uncompiled_gate.name:
                    raise AquaError('Gate mismatch at gate {0} ({1}, {2} params) of circuit against gate {3} ({4}, '
                                    '{5} params) of cached qobj'.format(cache_index,
                                                                 uncompiled_gate.name,
                                                                 len(uncompiled_gate.params),
                                                                 gate_num,
                                                                 compiled_gate.name,
                                                                 len(compiled_gate.params)))
                compiled_gate.params = np.array(uncompiled_gate.params, dtype=float).tolist()
        exec_qobj = copy.copy(self.qobjs[chunk])
        if self.skip_qobj_deepcopy: exec_qobj.experiments = self.qobjs[chunk].experiments[0:len(circuits)]
        else: exec_qobj.experiments = copy.deepcopy(self.qobjs[chunk].experiments[0:len(circuits)])

        if run_config is None:
            run_config = RunConfig(shots=1024, max_credits=10, memory=False)
        exec_qobj.config = QasmQobjConfig(**run_config.to_dict())
        exec_qobj.config.memory_slots = max(experiment.config.memory_slots for experiment in exec_qobj.experiments)
        exec_qobj.config.n_qubits = max(experiment.config.n_qubits for experiment in exec_qobj.experiments)
        return exec_qobj

    def clear_cache(self):
        self.qobjs = []
        self.mappings = []
        self.try_reusing_qobjs = True
