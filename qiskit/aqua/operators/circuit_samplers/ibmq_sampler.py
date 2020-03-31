# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Expectation Algorithm Base """

from typing import Dict, List, Optional
import logging

from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from ..operator_base import OperatorBase
from ..combo_operators import ListOp
from ..state_functions import StateFn, CircuitStateFn
from ..converters import DictToCircuitSum
from .circuit_sampler import CircuitSampler

logger = logging.getLogger(__name__)


class IBMQSampler(CircuitSampler):
    """ A sampler for remote IBMQ backends.

    TODO - make work.

    """

    def __init__(self,
                 backend: Optional[BaseBackend] = None,
                 quantum_instance: Optional[QuantumInstance] = None,
                 kwargs: Optional[Dict] = None) -> None:
        """
        Args:
            backend:
            quantum_instance:
            kwargs:
        """
        kwargs = {} if kwargs is None else kwargs
        self._qi = quantum_instance or QuantumInstance(backend=backend, **kwargs)

    def convert(self,
                operator: OperatorBase,
                params: dict = None):
        """ Accept the Operator and return the converted Operator """

        operator_dicts_replaced = DictToCircuitSum().convert(operator)
        reduced_op = operator_dicts_replaced.reduce()
        op_circuits = {}

        # pylint: disable=inconsistent-return-statements
        def extract_circuitstatefns(operator):
            if isinstance(operator, CircuitStateFn):
                op_circuits[str(operator)] = operator
            elif isinstance(operator, ListOp):
                for op in operator.oplist:
                    extract_circuitstatefns(op)
            else:
                return operator

        extract_circuitstatefns(reduced_op)
        sampled_statefn_dicts = self.sample_circuits(list(op_circuits.values()))

        def replace_circuits_with_dicts(operator):
            if isinstance(operator, CircuitStateFn):
                return sampled_statefn_dicts[str(operator)]
            elif isinstance(operator, ListOp):
                return operator.traverse(replace_circuits_with_dicts)
            else:
                return operator

        return replace_circuits_with_dicts(reduced_op)

    def sample_circuits(self,
                        op_circuits: Optional[List] = None,
                        param_bindings: Optional[List] = None) -> Dict:
        """
        Args:
            op_circuits: The list of circuits or CircuitStateFns to sample
            param_bindings: a list of parameter dictionaries to bind to each circuit.
        Returns:
            Dict: dictionary of sampled state functions
        """
        if all([isinstance(circ, CircuitStateFn) for circ in op_circuits]):
            circuits = [op_c.to_circuit(meas=True) for op_c in op_circuits]
        else:
            circuits = op_circuits

        results = self._qi.execute(circuits)
        sampled_statefn_dicts = {}
        for (op_c, circuit) in zip(op_circuits, circuits):
            # Taking square root because we're replacing a
            # statevector representation of probabilities.
            sqrt_counts = {b: (v * op_c.coeff / self._qi._run_config.shots) ** .5
                           for (b, v) in results.get_counts(circuit).items()}
            sampled_statefn_dicts[str(op_c)] = StateFn(sqrt_counts)
        return sampled_statefn_dicts
