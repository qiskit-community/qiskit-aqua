# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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

import numpy as np
from qiskit.dagcircuit import DAGCircuit

def summarize_circuits(circuits):
    """

    Summarize circuits based on DAGCircuit, and four metrics are summarized.
    Number of qubits and classical bits, and number of operations and depth of circuits.
    The average statistic is provided if multiple circuits are inputed.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    dag_circuits = [DAGCircuit.fromQuantumCircuit(circuit) for circuit in circuits]
    ret += "Submitting {} circuits.\n".format(len(dag_circuits))
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, dag_circuit in enumerate(dag_circuits):
        depth = dag_circuit.depth()
        width = dag_circuit.width()
        size = dag_circuit.size()
        classical_bits = dag_circuit.num_cbits()
        stats[0] += width
        stats[1] += classical_bits
        stats[2] += size
        stats[3] += depth
        ret += "{}-th circuit: {} qubits, {} classical bits and {} operations with depth {}\n".format(i, width, classical_bits, size, depth)
    if len(dag_circuits) > 1:
        stats /= len(dag_circuits)
        ret += "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} operations with depth {:.2f}\n".format(stats[0], stats[1], stats[2], stats[3])
    ret += "============================================================================\n"
    return ret