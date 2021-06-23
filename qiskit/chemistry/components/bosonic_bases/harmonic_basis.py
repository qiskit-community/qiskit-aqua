# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Bosonic Harmonic Basis """

from typing import Dict, List, Tuple, cast

import numpy as np

from qiskit.chemistry import WatsonHamiltonian
from .bosonic_basis import BosonicBasis


class HarmonicBasis(BosonicBasis):
    """Basis in which the Watson Hamiltonian is expressed.

    This class uses the Hermite polynomials (eigenstates of the harmonic oscillator) as a modal
    basis for the expression of the Watson Hamiltonian or any bosonic operator.

    References:

        [1] Ollitrault Pauline J., Chemical science 11 (2020): 6842-6855.

    """

    def __init__(self, watson_hamiltonian: WatsonHamiltonian, basis: List[int],
                 truncation_order: int = 3) -> None:
        """
        Args:
            watson_hamiltonian: A ``WatsonHamiltonian`` object which contains the hamiltonian
                information.
            basis: Is a list defining the number of modals per mode. E.g. for a 3 modes system
                with 4 modals per mode ``basis = [4, 4, 4]``.
            truncation_order: where is the Hamiltonian expansion truncation (1 for having only
                1-body terms, 2 for having on 1- and 2-body terms...)
        """

        self._watson = watson_hamiltonian
        self._basis = basis
        self._basis_size = max(basis)
        self._truncation_order = truncation_order

    @staticmethod
    def _harmonic_integrals(m: int, n: int, power: int, kinetic_term: bool = False) -> float:
        r"""Computes the integral of the Hamiltonian with the harmonic basis.

        This computation is as shown in [1].

        Args:
            m: first modal index
            n: second modal index
            power: the exponent on the coordinate (Q, Q^2, Q^3 or Q^4)
            kinetic_term: needs to be set to true to do the integral of the
                          kinetic part of the hamiltonian d^2/dQ^2

        Returns:
            The value of the integral.

        Raises:
            ValueError: If ``power`` is invalid

        References:

            [1] J. Chem. Phys. 135, 134108 (2011)
                https://doi.org/10.1063/1.3644895 (Table 1)

        """
        coeff = 0.0
        if power == 1:
            if m - n == 1:
                coeff = np.sqrt(m / 2)
        elif power == 2 and kinetic_term is True:
            if m - n == 0:
                coeff = -(m + 1 / 2)
            elif m - n == 2:
                coeff = np.sqrt(m * (m - 1)) / 2
            # coeff = -coeff
        elif power == 2 and kinetic_term is False:
            if m - n == 0:
                coeff = (m + 1 / 2)
            elif m - n == 2:
                coeff = np.sqrt(m * (m - 1)) / 2
        elif power == 3:
            if m - n == 1:
                coeff = 3 * np.power(m / 2, 3 / 2)
            elif m - n == 3:
                coeff = np.sqrt(m * (m - 1) * (m - 2)) / np.power(2, 3 / 2)
        elif power == 4:
            if m - n == 0:
                coeff = (6 * m * (m + 1) + 3) / 4
            elif m - n == 2:
                coeff = (m - 1 / 2) * np.sqrt(m * (m - 1))
            elif m - n == 4:
                coeff = np.sqrt(m * (m - 1) * (m - 2) * (m - 3)) / 4
        else:
            raise ValueError('The Q power is to high, only up to 4 is '
                             'currently supported.')
        return coeff * (np.sqrt(2) ** power)

    def _is_in_basis(self, indices, order, i):
        in_basis = True
        for j in range(order):
            for modal in [1, 2]:
                if indices[3 * j + modal][i] >= self._basis[indices[3 * j][i]]:
                    in_basis = False

        return in_basis

    def convert(self, threshold: float = 1e-6
                ) -> List[List[Tuple[List[List[int]], float]]]:
        """
        This prepares an array object representing a bosonic hamiltonian expressed
        in the harmonic basis. This object can directly be given to the BosonicOperator
        class to be mapped to a qubit hamiltonian.

        Args:
            threshold: the matrix elements of value below this threshold are discarded

        Returns:
            List of modes for input to creation of a bosonic hamiltonian in the harmonic basis

        Raises:
            ValueError: If problem with order value from computed modes
        """

        num_modes = len(self._basis)
        num_modals = self._basis_size

        harmonic_dict = {1: np.zeros((num_modes, num_modals, num_modals)),
                         2: np.zeros((num_modes, num_modals, num_modals,
                                      num_modes, num_modals, num_modals)),
                         3: np.zeros((num_modes, num_modals, num_modals,
                                      num_modes, num_modals, num_modals,
                                      num_modes, num_modals, num_modals))}

        for entry in self._watson.data:  # Entry is coeff (float) followed by indices (ints)
            coeff0 = cast(float, entry[0])
            indices = cast(List[int], entry[1:])

            kinetic_term = False

            # Note: these negative indices as detected below are explicitly generated in
            # _compute_modes for other potential uses. They are not wanted by this logic.
            if any(index < 0 for index in indices):
                kinetic_term = True
                indices = np.absolute(indices)  # type: ignore
            indexes = {}  # type: Dict[int, int]
            for i in indices:
                if indexes.get(i) is None:
                    indexes[i] = 1
                else:
                    indexes[i] += 1

            order = len(indexes.keys())
            modes = list(indexes.keys())

            if order == 1:
                for m in range(num_modals):
                    for n in range(m+1):

                        coeff = coeff0 * self._harmonic_integrals(
                            m, n, indexes[modes[0]], kinetic_term=kinetic_term)

                        if abs(coeff) > threshold:
                            harmonic_dict[1][modes[0]-1, m, n] += coeff
                            if m != n:
                                harmonic_dict[1][modes[0] - 1, n, m] += coeff

            elif order == 2:
                for m in range(num_modals):
                    for n in range(m+1):
                        coeff1 = coeff0 * self._harmonic_integrals(
                            m, n, indexes[modes[0]], kinetic_term=kinetic_term)
                        for j in range(num_modals):
                            for k in range(j+1):
                                coeff = coeff1 * self._harmonic_integrals(
                                    j, k, indexes[modes[1]], kinetic_term=kinetic_term)
                                if abs(coeff) > threshold:
                                    harmonic_dict[2][modes[0] - 1, m, n,
                                                     modes[1] - 1, j, k] += coeff
                                    if m != n:
                                        harmonic_dict[2][modes[0] - 1, n, m,
                                                         modes[1] - 1, j, k] += coeff
                                    if j != k:
                                        harmonic_dict[2][modes[0] - 1, m, n,
                                                         modes[1] - 1, k, j] += coeff
                                    if m != n and j != k:
                                        harmonic_dict[2][modes[0] - 1, n, m,
                                                         modes[1] - 1, k, j] += coeff
            elif order == 3:
                for m in range(num_modals):
                    for n in range(m+1):
                        coeff1 = coeff0 * self._harmonic_integrals(
                            m, n, indexes[modes[0]], kinetic_term=kinetic_term)
                        for j in range(num_modals):
                            for k in range(j+1):
                                coeff2 = coeff1 * self._harmonic_integrals(
                                    j, k, indexes[modes[1]], kinetic_term=kinetic_term)
                                # pylint: disable=locally-disabled, invalid-name
                                for p in range(num_modals):
                                    for q in range(p+1):
                                        coeff = coeff2 * self._harmonic_integrals(
                                            p, q, indexes[modes[2]], kinetic_term=kinetic_term)
                                        if abs(coeff) > threshold:
                                            harmonic_dict[3][modes[0] - 1, m, n,
                                                             modes[1] - 1, j, k,
                                                             modes[2] - 1, p, q] += coeff
                                            if m != n:
                                                harmonic_dict[3][modes[0] - 1, n, m,
                                                                 modes[1] - 1, j, k,
                                                                 modes[2] - 1, p, q] += coeff
                                            if k != j:
                                                harmonic_dict[3][modes[0] - 1, m, n,
                                                                 modes[1] - 1, k, j,
                                                                 modes[2] - 1, p, q] += coeff
                                            if p != q:
                                                harmonic_dict[3][modes[0] - 1, m, n,
                                                                 modes[1] - 1, j, k,
                                                                 modes[2] - 1, q, p] += coeff
                                            if m != n and k != j:
                                                harmonic_dict[3][modes[0] - 1, n, m,
                                                                 modes[1] - 1, k, j,
                                                                 modes[2] - 1, p, q] += coeff
                                            if m != n and p != q:
                                                harmonic_dict[3][modes[0] - 1, n, m,
                                                                 modes[1] - 1, j, k,
                                                                 modes[2] - 1, q, p] += coeff
                                            if p != q and k != j:
                                                harmonic_dict[3][modes[0] - 1, m, n,
                                                                 modes[1] - 1, k, j,
                                                                 modes[2] - 1, q, p] += coeff
                                            if m != n and j != k and p != q:
                                                harmonic_dict[3][modes[0] - 1, n, m,
                                                                 modes[1] - 1, k, j,
                                                                 modes[2] - 1, q, p] += coeff
            else:
                raise ValueError('Expansion of the PES is too large, only '
                                 'up to 3-body terms are supported')

        harmonics = []  # type: List[List[Tuple[List[List[int]], float]]]
        for idx in range(1, self._truncation_order + 1):
            all_indices = np.nonzero(harmonic_dict[idx])
            if len(all_indices[0]) != 0:
                harmonics.append([])
                values = harmonic_dict[idx][all_indices]
                for i in range(len(all_indices[0])):
                    if self._is_in_basis(all_indices, idx, i):
                        harmonics[- 1].append(([[all_indices[3 * j][i], all_indices[3 * j + 1][i],
                                                 all_indices[3 * j + 2][i]] for j in range(idx)],
                                               values[i]))

        return harmonics
