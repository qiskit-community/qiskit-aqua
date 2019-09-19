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
"""
This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
variational form.
For more information, see https://arxiv.org/abs/1805.04340
"""

import logging

import numpy as np

from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD

logger = logging.getLogger(__name__)


class UCCSDAdapt(UCCSD):
    """
        This trial wavefunction is an adapted version of the Unitary Coupled-Cluster
        Single and Double excitations variational form.
        It is used by the Adaptive VQE algorithm.

        Fur further details see:
            qiskit/chemistry/aqua_extensions/components/variational_forms/uccsd.py
            and qiskit/aqua/algorithms/adaptive/vqe_adapt/vqe_adapt.py
    """

    def __init__(self, *args, **kwargs):
        """Constructor.

        Args:
            Takes the same arguments as the UCCSD class. Please see
                qiskit/chemistry/aqua_extensions/components/variational_forms/uccsd.py
            for details.
         Raises:
             ValueError: Computed qubits do not match actual value
        """
        self.validate(locals())
        # construct as if this was an instance of UCCSD
        UCCSD.__init__(self, *args, **kwargs)

        # store full list of excitations as pool
        self._excitation_pool = self._hopping_ops.copy()

        # reset internal excitation list to be empty
        self._hopping_ops = []
        self._num_parameters = 0
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    @property
    def excitation_pool(self):
        """
        Getter of full list of available excitations (called the pool)
        Returns:
            list[WeightedPauliOperator]: excitation pool
        """
        return self._excitation_pool

    def _append_hopping_operator(self, excitation):
        """
        Registers a new hopping operator.
        """
        self._hopping_ops.append(excitation)
        self._num_parameters += 1
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def _pop_hopping_operator(self):
        """
        Pops the hopping operator that was added last.
        """
        self._hopping_ops.pop()
        self._num_parameters -= 1
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]
