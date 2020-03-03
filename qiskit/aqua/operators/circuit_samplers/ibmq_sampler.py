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

from . import CircuitSampler
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import OpVec, StateFn, StateFnCircuit
from qiskit.aqua.operators.converters import DicttoCircuitSum

logger = logging.getLogger(__name__)


class IBMQSampler(CircuitSampler):
    """ A sampler for remote IBMQ backends.

    """

    def __init__(self, backend, kwargs={}):
        """
        Args:
            backend():
        """
        self._backend = backend
        self._qi = QuantumInstance(backend=backend, **kwargs)

    def convert(self, operator):

        operator_dicts_replaced = DicttoCircuitSum().convert(operator)
        reduced_op = operator_dicts_replaced.reduce()
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
            op_circuits(list): The list of circuits or StateFnCircuits to sample
        """
        if all([isinstance(circ, StateFnCircuit) for circ in op_circuits]):
            circuits = [op_c.to_circuit(meas=True) for op_c in op_circuits]
        else:
            circuits = op_circuits

        results = self._qi.execute(circuits)
        sampled_statefn_dicts = {}
        for (op_c, circuit) in zip(op_circuits, circuits):
            # Taking square root because we're replacing a statevector representation of probabilities.
            sqrt_counts = {b: (v * op_c.coeff / self._qi._run_config.shots) ** .5
                           for (b, v) in results.get_counts(circuit).items()}
            sampled_statefn_dicts[str(op_c)] = StateFn(sqrt_counts)
        return sampled_statefn_dicts
