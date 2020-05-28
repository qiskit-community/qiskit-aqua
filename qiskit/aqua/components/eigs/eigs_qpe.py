# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Phase Estimation for getting the eigenvalues of a matrix."""

import warnings
from typing import Optional, List, Union
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit.aqua.circuits import PhaseEstimationCircuit
from qiskit.aqua.operators import LegacyBaseOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.components.iqfts import IQFT
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from .eigs import Eigenvalues

# pylint: disable=invalid-name


class EigsQPE(Eigenvalues):
    """Eigenvalues using Quantum Phase Estimation.

    Specifically, this class is based on PhaseEstimationCircuit with no measurements and
    has additional handling of negative eigenvalues, e.g. for :class:`~qiskit.aqua.algorithms.HHL`.
    It depends on :mod:`QFT <qiskit.aqua.components.qfts>` and
    :mod:`IQFT <qiskit.aqua.components.iqfts>` components.
    """

    def __init__(self,
                 operator: LegacyBaseOperator,
                 iqft: Union[QuantumCircuit, IQFT],
                 num_time_slices: int = 1,
                 num_ancillae: int = 1,
                 expansion_mode: str = 'trotter',
                 expansion_order: int = 1,
                 evo_time: Optional[float] = None,
                 negative_evals: bool = False,
                 ne_qfts: Optional[List] = None) -> None:
        """
        Args:
            operator: The Hamiltonian Operator object
            iqft: The Inverse Quantum Fourier Transform component
            num_time_slices: The number of time slices, has a minimum value of 1.
            num_ancillae: The number of ancillary qubits to use for the measurement,
                has a minimum value of 1.
            expansion_mode: The expansion mode ('trotter' | 'suzuki')
            expansion_order: The suzuki expansion order, has a minimum value of 1.
            evo_time: An optional evolution time which should scale the eigenvalue onto the range
                :math:`(0,1]` (or :math:`(-0.5,0.5]` for negative eigenvalues). Defaults to
                ``None`` in which case a suitably estimated evolution time is internally computed.
            negative_evals: Set ``True`` to indicate negative eigenvalues need to be handled
            ne_qfts: The QFT and IQFT components for handling negative eigenvalues
        """
        super().__init__()
        ne_qfts = ne_qfts if ne_qfts is not None else [None, None]
        validate_min('num_time_slices', num_time_slices, 1)
        validate_min('num_ancillae', num_ancillae, 1)
        validate_in_set('expansion_mode', expansion_mode, {'trotter', 'suzuki'})
        validate_min('expansion_order', expansion_order, 1)
        self._operator = op_converter.to_weighted_pauli_operator(operator)

        if isinstance(iqft, IQFT):
            warnings.warn('Providing a qiskit.aqua.components.iqfts.IQFT module as `iqft` argument '
                          'to HHL is deprecated as of 0.7.0 and will be removed no earlier than '
                          '3 months after the release. '
                          'You should pass a QuantumCircuit instead, see '
                          'qiskit.circuit.library.QFT and the .inverse() method.',
                          DeprecationWarning, stacklevel=2)
        self._iqft = iqft

        self._num_ancillae = num_ancillae
        self._num_time_slices = num_time_slices
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._evo_time = evo_time
        self._negative_evals = negative_evals

        if ne_qfts and any(isinstance(ne_qft, IQFT) for ne_qft in ne_qfts):
            warnings.warn('Providing a qiskit.aqua.components.iqfts.IQFT module in the `ne_qft` '
                          'argument to HHL is deprecated as of 0.7.0 and will be removed no '
                          'earlier than 3 months after the release. '
                          'You should pass a QuantumCircuit instead, see '
                          'qiskit.circuit.library.QFT and the .inverse() method.',
                          DeprecationWarning, stacklevel=2)
        self._ne_qfts = ne_qfts

        self._circuit = None
        self._output_register = None
        self._input_register = None
        self._init_constants()

    def _init_constants(self):
        # estimate evolution time
        if self._evo_time is None:
            lmax = sum([abs(p[0]) for p in self._operator.paulis])
            if not self._negative_evals:
                self._evo_time = (1 - 2 ** -self._num_ancillae) * 2 * np.pi / lmax
            else:
                self._evo_time = (1 / 2 - 2 ** -self._num_ancillae) * 2 * np.pi / lmax

        # check for identify paulis to get its coef for applying global
        # phase shift on ancillae later
        num_identities = 0
        for p in self._operator.paulis:
            if np.all(p[1].z == 0) and np.all(p[1].x == 0):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                self._ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]

    def get_register_sizes(self):
        return self._operator.num_qubits, self._num_ancillae

    def get_scaling(self):
        return self._evo_time

    def construct_circuit(self, mode, register=None):
        """Construct the eigenvalues estimation using the PhaseEstimationCircuit

        Args:
            mode (str): construction mode, 'matrix' not supported
            register (QuantumRegister): the register to use for the quantum state

        Returns:
            QuantumCircuit: object for the constructed circuit
        Raises:
            ValueError: QPE is only possible as a circuit not as a matrix
        """

        if mode == 'matrix':
            raise ValueError('QPE is only possible as a circuit not as a matrix.')

        pe = PhaseEstimationCircuit(
            operator=self._operator, state_in=None, iqft=self._iqft,
            num_time_slices=self._num_time_slices, num_ancillae=self._num_ancillae,
            expansion_mode=self._expansion_mode, expansion_order=self._expansion_order,
            evo_time=self._evo_time
        )

        a = QuantumRegister(self._num_ancillae)
        q = register

        qc = pe.construct_circuit(state_register=q, ancillary_register=a)

        # handle negative eigenvalues
        if self._negative_evals:
            self._handle_negative_evals(qc, a)

        self._circuit = qc
        self._output_register = a
        self._input_register = q
        return self._circuit

    def _handle_negative_evals(self, qc, q):
        sgn = q[0]
        qs = [q[i] for i in range(1, len(q))]

        def apply_ne_qft(ne_qft):
            if isinstance(ne_qft, QuantumCircuit):
                # check if QFT has the right size
                if ne_qft.num_qubits != len(qs):
                    try:  # try resizing
                        ne_qft.num_qubits = len(qs)
                    except AttributeError:
                        raise ValueError('The IQFT cannot be resized and does not have the '
                                         'required size of {}'.format(len(qs)))

                if hasattr(ne_qft, 'do_swaps'):
                    ne_qft.do_swaps = False
                qc.append(ne_qft.to_instruction(), qs)
            else:
                ne_qft.construct_circuit(mode='circuit', qubits=qs, circuit=qc, do_swaps=False)

        for qi in qs:
            qc.cx(sgn, qi)
        apply_ne_qft(self._ne_qfts[0])
        for i, qi in enumerate(reversed(qs)):
            qc.cu1(2 * np.pi / 2 ** (i + 1), sgn, qi)
        apply_ne_qft(self._ne_qfts[1])
