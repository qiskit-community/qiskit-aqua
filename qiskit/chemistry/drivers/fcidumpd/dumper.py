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

"""FCIDump dumper."""

from typing import List, Optional, Union, TextIO, Tuple
import itertools
import numpy as np


def dump(outpath: str, norb: int, nelec: int, hijs: np.ndarray, hijkls: np.ndarray, einact: float,
         ms2: int = 0, orbsym: Optional[List[str]] = None, isym: int = 1
         ) -> None:
    # pylint: disable=wrong-spelling-in-docstring
    """Generates a FCIDump output.

    Args:
        outpath: Path to the output file.
        norb: The number of orbitals.
        nelec: The number of electrons.
        hijs: The pair of alpha and beta 1-electron integrals. The latter may be None.
        hijkls: The triplet of alpha/alpha, beta/alpha and beta/beta 2-electron integrals. The
            latter two may be None.
        einact: The inactive energy.
        ms2: 2*S, where S is the spin quantum number.
        orbsym: A list of spatial symmetries of the orbitals.
        isym: The spatial symmetry of the wave function.
    """
    hij, hij_b = hijs
    hijkl, hijkl_ba, hijkl_bb = hijkls
    # assert that either all beta variables are None or all of them are not
    assert all(h is None for h in [hij_b, hijkl_ba, hijkl_bb]) \
        or all(h is not None for h in [hij_b, hijkl_ba, hijkl_bb])
    assert norb == hij.shape[0] == hijkl.shape[0]
    mos = range(norb)
    with open(outpath, 'w') as outfile:
        # print header
        outfile.write('&FCI NORB={:4d},NELEC={:4d},MS2={:4d}\n'.format(norb, nelec, ms2))
        if orbsym is None:
            outfile.write(' ORBSYM=' + '1,'*norb + '\n')
        else:
            assert len(orbsym) == norb
            outfile.write(' ORBSYM=' + ','.join(orbsym) + '\n')
        outfile.write(' ISYM={:d},\n&END\n'.format(isym))
        # append 2e integrals
        _dump_2e_ints(hijkl, mos, outfile)
        if hijkl_ba is not None:
            _dump_2e_ints(hijkl_ba.transpose(), mos, outfile, beta=1)
        if hijkl_bb is not None:
            _dump_2e_ints(hijkl_bb, mos, outfile, beta=2)
        # append 1e integrals
        _dump_1e_ints(hij, mos, outfile)
        if hij_b is not None:
            _dump_1e_ints(hij_b, mos, outfile, beta=True)
        # TODO append MO energies (last three indices are 0)
        # append inactive energy
        _write_to_outfile(outfile, einact, (0, 0, 0, 0))


def _dump_1e_ints(hij: List[float],
                  mos: Union[range, List[int]],
                  outfile: TextIO,
                  beta: bool = False) -> None:
    idx_offset = 1 if not beta else 1+len(mos)
    hij_elements = set()
    for i, j in itertools.product(mos, repeat=2):
        if i == j:
            _write_to_outfile(outfile, hij[i][j], (i+idx_offset, j+idx_offset, 0, 0))
            continue
        if (j, i) in hij_elements and np.isclose(hij[i][j], hij[j][i]):
            continue
        _write_to_outfile(outfile, hij[i][j], (i+idx_offset, j+idx_offset, 0, 0))
        hij_elements.add((i, j))


def _dump_2e_ints(hijkl: np.ndarray,
                  mos: Union[range, List[int]],
                  outfile: TextIO,
                  beta: int = 0) -> None:
    idx_offsets = [1, 1]
    for b in range(beta):
        idx_offsets[1-b] += len(mos)
    hijkl_elements = set()
    # pylint: disable=invalid-name
    for elem in itertools.product(mos, repeat=4):
        if np.isclose(hijkl[elem], 0.0, atol=1e-14):
            continue
        if len(set(elem)) == 1:
            _write_to_outfile(outfile, hijkl[elem], (*[e+idx_offsets[0] for e in elem[:2]],
                                                     *[e+idx_offsets[1] for e in elem[2:]]))
            continue
        if beta != 1 and elem[::-1] in hijkl_elements and \
                np.isclose(hijkl[elem], hijkl[elem[::-1]]):
            continue
        bra_perms = set(itertools.permutations(elem[:2]))
        ket_perms = set(itertools.permutations(elem[2:]))
        if beta == 1:
            permutations = itertools.product(bra_perms, ket_perms)
        else:
            permutations = itertools.chain(
                itertools.product(bra_perms, ket_perms),
                itertools.product(ket_perms, bra_perms)
            )
        for perm in {e1 + e2 for e1, e2 in permutations}:
            if perm in hijkl_elements and np.isclose(hijkl[elem], hijkl[perm]):
                break
        else:
            _write_to_outfile(outfile, hijkl[elem], (*[e+idx_offsets[0] for e in elem[:2]],
                                                     *[e+idx_offsets[1] for e in elem[2:]]))
            hijkl_elements.add(elem)


def _write_to_outfile(outfile: TextIO, value: float, indices: Tuple):
    outfile.write('{:23.16E}{:4d}{:4d}{:4d}{:4d}\n'.format(value, *indices))
