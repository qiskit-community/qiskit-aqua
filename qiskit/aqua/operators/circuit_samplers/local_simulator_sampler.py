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
from abc import abstractmethod

from qiskit import BasicAer

from . import CircuitSampler
from qiskit.aqua import AquaError, QuantumAlgorithm, QuantumInstance
from qiskit.aqua.operators import OpVec, StateFn, StateFnCircuit

logger = logging.getLogger(__name__)


class LocalSimulatorSampler(CircuitSampler):
    """ A sampler for local Quantum simulator backends.

    """

    def __init__(self, backend, hw_backend_to_emulate=None, kwargs={}):
        """
        Args:
            backend():
            hw_backend_to_emulate():
        """
        self._backend = backend
        if hw_backend_to_emulate and has_aer and 'noise_model' not in kwargs:
            from qiskit.providers.aer.noise import NoiseModel
            # TODO figure out Aer versioning
            kwargs['noise_model'] = NoiseModel.from_backend(hw_backend_to_emulate)
        self._qi = QuantumInstance(backend=backend, **kwargs)

    def convert(self, operator):

        reduced_op = operator.reduce()
        op_circuits = {}

        def extract_statefncircuits(operator):
            if isinstance(operator, StateFnCircuit):
                op_circuits[str(operator)] = operator
            elif isinstance(operator, OpVec):
                for op in operator.oplist:
                    extract_statefncircuits(op)
            else:
                return operator

        extract_statefncircuits(reduced_op)
        sampled_statefn_dicts = self.sample_circuits(list(op_circuits.values()))

        def replace_circuits_with_dicts(operator):
            if isinstance(operator, StateFnCircuit):
                return sampled_statefn_dicts[str(operator)]
            elif isinstance(operator, OpVec):
                return operator.traverse(replace_circuits_with_dicts)
            else:
                return operator

        return replace_circuits_with_dicts(reduced_op)

    def sample_circuits(self, op_circuits):
        """
        Args:
            op_circuits(list): The list of circuits to sample
        """
        circuits = [op_c.to_circuit(meas=True) for op_c in op_circuits]
        results = self._qi.execute(circuits)
        sampled_statefn_dicts = {}
        # TODO it's annoying that I can't just reuse the logic to create a StateFnDict from a results object here.
        for (op_c, circuit) in zip(op_circuits, circuits):
            sampled_statefn_dicts[str(op_c)] = StateFn(results.get_counts(circuit)) * \
                                               (1/self._qi._run_config.shots)
        return sampled_statefn_dicts
