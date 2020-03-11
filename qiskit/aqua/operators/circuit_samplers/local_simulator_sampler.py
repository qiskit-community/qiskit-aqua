# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm Base """

import logging
import numpy as np
from functools import partial

from . import CircuitSampler
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import OpVec, StateFn, StateFnCircuit, Zero
from qiskit.aqua.operators.converters import DicttoCircuitSum

from qiskit.aqua.utils.backend_utils import is_aer_provider, is_statevector_backend, is_aer_qasm

logger = logging.getLogger(__name__)


class LocalSimulatorSampler(CircuitSampler):
    """ A sampler for local Quantum simulator backends.
    # TODO replace OpMatrices with Terra unitary gates to make them runnable. Note, Terra's Operator has a
    to_instruction method.
    """

    def __init__(self, backend=None, hw_backend_to_emulate=None, kwargs={}, statevector=False, snapshot=False):
        """
        Args:
            backend():
            hw_backend_to_emulate():
        """
        if hw_backend_to_emulate and is_aer_provider(backend) and 'noise_model' not in kwargs:
            from qiskit.providers.aer.noise import NoiseModel
            # TODO figure out Aer versioning
            kwargs['noise_model'] = NoiseModel.from_backend(hw_backend_to_emulate)

        self._qi = backend if isinstance(backend, QuantumInstance) else QuantumInstance(backend=backend, **kwargs)
        self._last_op = None
        self._reduced_op_cache = None
        self._circuit_ops_cache = None
        self._transpiled_circ_cache = None
        self._statevector = statevector
        if self._statevector and not is_statevector_backend(self.quantum_instance.backend):
            raise ValueError('Statevector mode for circuit sampling requires statevector '
                             'backend, not {}.'.format(backend))
        self._snapshot = snapshot
        if self._snapshot and not is_aer_qasm(self.quantum_instance.backend):
            raise ValueError('Snapshot mode for expectation values requires Aer qasm '
                             'backend, not {}.'.format(backend))

    @property
    def backend(self):
        return self.quantum_instance.backend

    @backend.setter
    def backend(self, backend):
        self.quantum_instance = QuantumInstance(backend=backend, **kwargs)

    @property
    def quantum_instance(self):
        return self._qi

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance):
        self._qi = quantum_instance

    def convert(self, operator, params=None):
        if self._last_op is None or not operator == self._last_op:
            # Clear caches
            self._last_op = operator
            self._reduced_op_cache = None
            self._circuit_ops_cache = None
            self._transpiled_circ_cache = None

        if not self._reduced_op_cache:
            operator_dicts_replaced = DicttoCircuitSum().convert(operator)
            self._reduced_op_cache = operator_dicts_replaced.reduce()

        if not self._circuit_ops_cache:
            self._circuit_ops_cache = {}

            def extract_statefncircuits(operator):
                if isinstance(operator, StateFnCircuit):
                    self._circuit_ops_cache[id(operator)] = operator
                elif isinstance(operator, OpVec):
                    for op in operator.oplist:
                        extract_statefncircuits(op)
                else:
                    return operator

            extract_statefncircuits(self._reduced_op_cache)

        if params:
            num_parameterizations = len(list(params.values())[0])
            param_bindings = [{param: value_list[i] for (param, value_list) in params.items()}
                              for i in range(num_parameterizations)]
        else:
            param_bindings = None
            num_parameterizations = 1

        # Don't pass circuits if we have in the cache the sampling function knows to use the cache.
        circs = list(self._circuit_ops_cache.values()) if not self._transpiled_circ_cache else None
        sampled_statefn_dicts = self.sample_circuits(op_circuits=circs, param_bindings=param_bindings)

        def replace_circuits_with_dicts(operator, param_index=0):
            if isinstance(operator, StateFnCircuit):
                return sampled_statefn_dicts[id(operator)][param_index]
            elif isinstance(operator, OpVec):
                return operator.traverse(partial(replace_circuits_with_dicts, param_index=param_index))
            else:
                return operator

        if params:
            return OpVec([replace_circuits_with_dicts(self._reduced_op_cache, param_index=i)
                          for i in range(num_parameterizations)])
        else:
            return replace_circuits_with_dicts(self._reduced_op_cache, param_index=0)

    def sample_circuits(self, op_circuits=None, param_bindings=None):
        """
        Args:
            op_circuits(list): The list of circuits or StateFnCircuits to sample
        """
        if op_circuits or not self._transpiled_circ_cache:
            if all([isinstance(circ, StateFnCircuit) for circ in op_circuits]):
                if self._statevector:
                    circuits = [op_c.to_circuit(meas=False) for op_c in op_circuits]
                else:
                    circuits = [op_c.to_circuit(meas=True) for op_c in op_circuits]
            else:
                circuits = op_circuits
            self._transpiled_circ_cache = self._qi.transpile(circuits)
        else:
            op_circuits = list(self._circuit_ops_cache.values())

        if param_bindings is not None:
            ready_circs = [circ.bind_parameters(binding)
                           for circ in self._transpiled_circ_cache for binding in param_bindings]
        else:
            ready_circs = self._transpiled_circ_cache

        results = self._qi.execute(ready_circs, had_transpiled=True)

        sampled_statefn_dicts = {}
        for i, op_c in enumerate(op_circuits):
            # Taking square root because we're replacing a statevector representation of probabilities.
            reps = len(param_bindings) if param_bindings is not None else 1
            c_statefns = []
            for j in range(reps):
                circ_index = (i*reps) + j
                if self._statevector:
                    result_sfn = StateFn(op_c.coeff * results.get_statevector(circ_index))
                elif self._snapshot:
                    snapshot_data = results.data(circ_index)['snapshots']
                    avg = snapshot_data['expectation_value']['expval'][0]['value']
                    if isinstance(avg, (list, tuple)):
                        # Aer versions before 0.4 use a list snapshot format
                        # which must be converted to a complex value.
                        avg = avg[0] + 1j * avg[1]
                    # Will be replaced with just avg when eval is called later
                    num_qubits = op_circuits[0].num_qubits
                    result_sfn = (Zero^num_qubits).adjoint() * avg
                else:
                    result_sfn = StateFn({b: (v * op_c.coeff / self._qi._run_config.shots) ** .5
                                          for (b, v) in results.get_counts(circ_index).items()})
                c_statefns.append(result_sfn)
            sampled_statefn_dicts[id(op_c)] = c_statefns
        return sampled_statefn_dicts
