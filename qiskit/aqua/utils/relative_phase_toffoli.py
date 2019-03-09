# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
"""
Relative Phase Toffoli Gates.
"""

from qiskit import QuantumCircuit
from qiskit.qasm import pi


def _apply_rccx(circ, a, b, c):
    circ.u2(0, pi, c)  # h
    circ.u1(pi / 4, c)  # t
    circ.cx(b, c)
    circ.u1(-pi / 4, c)  # tdg
    circ.cx(a, c)
    circ.u1(pi / 4, c)  # t
    circ.cx(b, c)
    circ.u1(-pi / 4, c)  # tdg
    circ.u2(0, pi, c)  # h


def _apply_rcccx(circ, a, b, c, d):
    circ.u2(0, pi, d)  # h
    circ.u1(pi / 4, d)  # t
    circ.cx(c, d)
    circ.u1(-pi / 4, d)  # tdg
    circ.u2(0, pi, d)  # h
    circ.cx(a, d)
    circ.u1(pi / 4, d)  # t
    circ.cx(b, d)
    circ.u1(-pi / 4, d)  # tdg
    circ.cx(a, d)
    circ.u1(pi / 4, d)  # t
    circ.cx(b, d)
    circ.u1(-pi / 4, d)  # tdg
    circ.u2(0, pi, d)  # h
    circ.u1(pi / 4, d)  # t
    circ.cx(c, d)
    circ.u1(-pi / 4, d)  # tdg
    circ.u2(0, pi, d)  # h


def rccx(self, ctl1, ctl2, tgt):
    """
    Apply Relative Phase Toffoli from ctl1 and ctl2 to tgt.

    https://arxiv.org/pdf/1508.03273.pdf Figure 3
    """
    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, tgt])
    _apply_rccx(self, ctl1, ctl2, tgt)


def rcccx(self, ctl1, ctl2, ctl3, tgt):
    """
    Apply 3-Control Relative Phase Toffoli from ctl1, ctl2, and ctl3 to tgt.

    https://arxiv.org/pdf/1508.03273.pdf Figure 4
    """
    self._check_qubit(ctl1)
    self._check_qubit(ctl2)
    self._check_qubit(ctl3)
    self._check_qubit(tgt)
    self._check_dups([ctl1, ctl2, ctl3, tgt])
    _apply_rcccx(self, ctl1, ctl2, ctl3, tgt)


QuantumCircuit.rccx = rccx
QuantumCircuit.rcccx = rcccx
