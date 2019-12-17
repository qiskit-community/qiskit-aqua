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

""" Q Factory """

from qiskit.aqua.utils import CircuitFactory

from qiskit.aqua.algorithms.single_sample.amplitude_estimation.s_psi_0_reflection \
    import SPsi0Factory
from qiskit.aqua.algorithms.single_sample.amplitude_estimation.s_0_reflection import S0Factory

# pylint: disable=invalid-name


class QFactory(CircuitFactory):
    """ Q Factory """
    def __init__(self, a_factory, i_objective):

        super().__init__(a_factory.num_target_qubits)

        # store A factory
        self.a_factory = a_factory
        self.i_objective = i_objective

        # construct reflection factories
        self.s_psi_0_reflection_factory = SPsi0Factory(a_factory.num_target_qubits, i_objective)
        self.s_0_reflection_factory = S0Factory(a_factory.num_target_qubits)

        # determine number of required ancillas (A does not need to be controlled within Q!)
        self._num_a_ancillas = a_factory.required_ancillas()
        self._num_s_psi_0_ancillas = self.s_psi_0_reflection_factory.required_ancillas()
        self._num_s_psi_0_ancillas_controlled = \
            self.s_psi_0_reflection_factory.required_ancillas_controlled()
        self._num_s_0_ancillas = self.s_0_reflection_factory.required_ancillas()
        self._num_s_0_ancillas_controlled = \
            self.s_0_reflection_factory.required_ancillas_controlled()
        self._num_ancillas = max(self._num_s_psi_0_ancillas,
                                 self._num_s_0_ancillas, self._num_a_ancillas)
        self._num_ancillas_controlled = max(self._num_s_psi_0_ancillas_controlled,
                                            self._num_s_0_ancillas_controlled, self._num_a_ancillas)

        # determine total number of qubits
        self._num_qubits = a_factory.num_target_qubits + self._num_ancillas
        self._num_qubits_controlled = a_factory.num_target_qubits + self._num_ancillas_controlled

    def required_ancillas(self):
        return self._num_ancillas

    def required_ancillas_controlled(self):
        return self._num_ancillas_controlled

    def build(self, qc, q, q_ancillas=None, params=None):
        self.s_psi_0_reflection_factory.build(qc, q, q_ancillas)
        self.a_factory.build_inverse(qc, q, q_ancillas)
        self.s_0_reflection_factory.build(qc, q, q_ancillas)
        self.a_factory.build(qc, q, q_ancillas)

    def build_controlled(self, qc, q, q_control, q_ancillas=None, use_basis_gates=True):
        # A operators do not need to be controlled, since they cancel out in the not-controlled case
        self.s_psi_0_reflection_factory.build_controlled(qc,
                                                         q, q_control, q_ancillas, use_basis_gates)
        self.a_factory.build_inverse(qc, q, q_ancillas)
        self.s_0_reflection_factory.build_controlled(qc, q, q_control, q_ancillas, use_basis_gates)
        self.a_factory.build(qc, q, q_ancillas)
