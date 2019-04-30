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

from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD


class MP2Info:
    """A utility class for Moller-Plesset 2nd order (MP2) information

    Each double excitation given by [i,a,j,b] has a coefficient computed using
      coeff = -(2 * Tiajb - Tibja)/(oe[b] + oe[a] - oe[i] - oe[j])
    where oe[] is the orbital energy

    and an energy delta given by
      e_delta = coeff * Tiajb

    All the computations are done using the molecule orbitals but the indexes used
    in the excitation information passed in and out are in the block spin orbital
    numbering as normally used by the chemistry stack.
    """

    def __init__(self, qmolecule, threshold=1e-12):
        """
        A utility class for MP2 info

        Args:
            qmolecule (QMolecule): QMolecule from chemistry driver
            threshold (float): Computed coefficients and energy deltas will be set to
                               zero if their value is below this threshold
        """
        self._terms, self._mp2_delta = _compute_mp2(qmolecule, threshold)
        self._mp2_energy = qmolecule.hf_energy + self._mp2_delta
        self._num_orbitals = qmolecule.num_orbitals
        self._core_orbitals = qmolecule.core_orbitals

    @property
    def mp2_delta(self):
        """
        Get the MP2 delta energy correction for the molecule

        Returns:
             float: The MP2 delta energy
        """
        return self._mp2_delta

    @property
    def mp2_energy(self):
        """
        Get the MP2 energy for the molecule

        Returns:
            float: The MP2 energy
        """
        return self._mp2_energy

    def mp2_terms(self, freeze_core=False, orbital_reduction=None):
        """
        Gets the set of MP2 terms for the molecule taking into account index adjustments
        due to frozen core and/or other orbital reduction

        Args:
            freeze_core (bool): Whether core orbitals are frozen or not
            orbital_reduction (list): An optional list of ints indicating removed orbitals

        Returns:
            dict: A dictionary of excitations where the key is a string in the form
                  from_to_from_to e.g. 0_4_6_10 and the value is a tuple of
                  (coeff, e_delta)
        """
        orbital_reduction = orbital_reduction if orbital_reduction is not None else []

        # Compute the list of orbitals that will be removed. Here we do not care whether
        # it is occupied or not since the goal will be to subset the full set of excitation
        # terms, we originally computed, down to the set that exist within the remaining
        # orbitals.
        core_list = self._core_orbitals if freeze_core else []
        reduce_list = orbital_reduction
        reduce_list = [x + self._num_orbitals if x < 0 else x for x in reduce_list]
        remove_orbitals = sorted(set(core_list).union(set(reduce_list)))
        remove_spin_orbitals = remove_orbitals + [x + self._num_orbitals for x in remove_orbitals]

        # An array of original indexes of the full set of spin orbitals. Plus an
        # array which will end up having the new indexes at the corresponding positions
        # of the original orbital after the removal has taken place. The original full
        # set will correspondingly have -1 values entered where orbitals have been removed
        full_spin_orbs = [*range(0, 2 * self._num_orbitals)]
        remain_spin_orbs = [-1] * len(full_spin_orbs)

        new_idx = 0
        for i in range(len(full_spin_orbs)):
            if full_spin_orbs[i] in remove_spin_orbitals:
                full_spin_orbs[i] = -1
                continue
            remain_spin_orbs[i] = new_idx
            new_idx += 1

        # Now we look through all the original excitations and check if all the from and to
        # values in the set or orbitals exists (is a subset of) the remaining orbitals in the
        # full spin set (note this now has -1 as value in indexes for which the orbital was
        # removed. If its a subset we remap the orbitals to the values that correspond to the
        # remaining spin orbital indexes.
        ret_terms = {}
        for k, v in self._terms.items():
            orbs = _str_to_list(k)
            if set(orbs) < set(full_spin_orbs):
                new_idxs = [remain_spin_orbs[elem] for elem in orbs]
                coeff, e_delta = v
                ret_terms[_list_to_str(new_idxs)] = (coeff, e_delta)

        return ret_terms

    def mp2_get_term_info(self, excitation_list, freeze_core=False, orbital_reduction=None):
        """
        With a reduced active space the set of used excitations can be less than allowing
        all available excitations. Given a (sub)set of excitations in the space this will return
        a list of correlation coefficients and a list of correlation energies ordered as per
        the excitation list provided.

        Args:
            excitation_list (list): A list of excitations for which to get the coeff and e_delta
            freeze_core (bool): Whether core orbitals are frozen or not
            orbital_reduction (list): An optional list of ints indicating removed orbitals

        Returns:
            list, list: List of coefficients and list of energy deltas
        """
        terms = self.mp2_terms(freeze_core, orbital_reduction)
        coeffs = []
        e_deltas = []
        for excitation in excitation_list:
            if len(excitation) != 4:
                raise ValueError('Excitation entry must be of length 4')
            key = _list_to_str(excitation)
            if key in terms:
                coeff, e_delta = terms[key]
                coeffs.append(coeff)
                e_deltas.append(e_delta)
            else:
                raise ValueError('Excitation {} not present in mp2 terms'.format(excitation))
        return coeffs, e_deltas


def _list_to_str(idxs):
    return '_'.join([str(x) for x in idxs])


def _str_to_list(str_idxs):
    return [int(x) for x in str_idxs.split('_')]


def _compute_mp2(qmolecule, threshold):
    terms = {}
    mp2_delta = 0

    num_particles = qmolecule.num_alpha + qmolecule.num_beta
    num_orbitals = qmolecule.num_orbitals
    ints = qmolecule.mo_eri_ints
    oe = qmolecule.orbital_energies

    # Orbital indexes given by this method are numbered according to the blocked spin ordering
    singles, doubles = UCCSD.compute_excitation_lists(num_particles, num_orbitals * 2, same_spin_doubles=True)

    # doubles is list of [from, to, from, to] in spin orbital indexing where alpha runs
    # from 0 to num_orbitals-1, and beta from num_orbitals to num_orbitals*2-1
    for n in range(len(doubles)):
        idxs = doubles[n]
        i = idxs[0] % num_orbitals  # Since spins are same drop to MO indexing
        j = idxs[2] % num_orbitals
        a = idxs[1] % num_orbitals
        b = idxs[3] % num_orbitals

        tiajb = ints[i, a, j, b]
        tibja = ints[i, b, j, a]

        num = (2 * tiajb - tibja)
        denom = oe[b] + oe[a] - oe[i] - oe[j]
        coeff = -num / denom
        coeff = coeff if abs(coeff) > threshold else 0
        e_delta = coeff * tiajb
        e_delta = e_delta if abs(e_delta) > threshold else 0

        terms[_list_to_str(idxs)] = (coeff, e_delta)
        mp2_delta += e_delta

    return terms, mp2_delta
