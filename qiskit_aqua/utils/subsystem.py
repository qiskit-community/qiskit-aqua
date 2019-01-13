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

import numpy as np
from scipy.linalg import sqrtm
from collections import defaultdict

from qiskit.tools.qi.qi import partial_trace


def get_subsystem_statevector(statevector, trace_systems):
    # trace system is a list of qubits one wants to trace. E.g.
    # to trace qubits 0 and 4 trace_systems = [0,4]
    rho = np.outer(statevector, statevector)
    rho_sub = partial_trace(rho, trace_systems)
    u, s, v = np.linalg.svd(rho_sub)
    state_sub = np.transpose(np.conj(np.dot(u, s)))
    return state_sub


def get_subsystem_fidelity(statevector, trace_systems, subsystem_state):
    rho = np.outer(np.conj(statevector), statevector)
    rho_sub = partial_trace(rho, trace_systems)
    rho_sub_in = np.outer(np.conj(subsystem_state), subsystem_state)
    fidelity = np.trace(
        sqrtm(
            np.dot(
                np.dot(sqrtm(rho_sub), rho_sub_in),
                sqrtm(rho_sub)
            )
        )
    ) ** 2
    return fidelity


def get_subsystems_counts(entire_system_counts):
    mixed_measurements = list(entire_system_counts)
    subsystems_counts = [defaultdict(int) for _ in mixed_measurements[0].split()]
    for mixed_measurement in mixed_measurements:
        count = entire_system_counts[mixed_measurement]
        for k, d in zip(mixed_measurement.split(), subsystems_counts):
            d[k] += count
    return [dict(d) for d in subsystems_counts]
