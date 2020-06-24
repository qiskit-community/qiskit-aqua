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

"""FCIDump parser."""

from typing import Any, Dict, Set, Tuple
import itertools
import re
import numpy as np

from qiskit.chemistry import QiskitChemistryError


def parse(fcidump: str) -> Dict[str, Any]:
    # pylint: disable=wrong-spelling-in-comment
    """Parses a FCIDump output.

    Args:
        fcidump: Path to the FCIDump file.
    Raises:
        QiskitChemistryError: If the input file cannot be found, if a required field in the FCIDump
            file is missing, if wrong integral indices are encountered, or if the alpha/beta or
            beta/alpha 2-electron integrals are mixed.
    Returns:
        A dictionary storing the parsed data.
    """
    try:
        with open(fcidump, 'r') as file:
            fcidump_str = file.read()
    except OSError as ex:
        raise QiskitChemistryError("Input file '{}' cannot be read!".format(fcidump)) from ex

    output = {}  # type: Dict[str, Any]

    # FCIDump starts with a Fortran namelist of meta data
    namelist_end = re.search('(/|&END)', fcidump_str)
    metadata = fcidump_str[:namelist_end.start(0)]
    metadata = ' '.join(metadata.split())  # replace duplicate whitespace and newlines
    # we know what elements to look for so we don't get too fancy with the parsing
    # pattern explanation:
    #  .*?      any text
    #  (*|*),   match either part of this group followed by a comma
    #  [-+]?    match up to a single - or +
    #  \d*.\d+  number format
    pattern = r'.*?([-+]?\d*\.\d+|[-+]?\d+),'
    # we parse the values in the order in which they are listed in Knowles1989
    _norb = re.search('NORB'+pattern, metadata)
    if _norb is None:
        raise QiskitChemistryError("The required NORB entry of the FCIDump format is missing!")
    norb = int(_norb.groups()[0])
    output['NORB'] = norb
    _nelec = re.search('NELEC'+pattern, metadata)
    if _nelec is None:
        raise QiskitChemistryError("The required NELEC entry of the FCIDump format is missing!")
    output['NELEC'] = int(_nelec.groups()[0])
    # the rest of these values may occur and are set to their defaults otherwise
    _ms2 = re.search('MS2'+pattern, metadata)
    output['MS2'] = int(_ms2.groups()[0]) if _ms2 else 0
    _isym = re.search('ISYM'+pattern, metadata)
    output['ISYM'] = int(_isym.groups()[0]) if _isym else 1
    # ORBSYM holds a list, thus it requires a little different treatment
    _orbsym = re.search(r'ORBSYM.*?'+r'(\d+),'*norb, metadata)
    output['ORBSYM'] = [int(s) for s in _orbsym.groups()] if _orbsym else [1] * norb
    _iprtim = re.search('IPRTIM'+pattern, metadata)
    output['IPRTIM'] = int(_iprtim.groups()[0]) if _iprtim else -1
    _int = re.search('INT'+pattern, metadata)
    output['INT'] = int(_int.groups()[0]) if _int else 5
    _memory = re.search('MEMORY'+pattern, metadata)
    output['MEMORY'] = int(_memory.groups()[0]) if _memory else 10000
    _core = re.search('CORE'+pattern, metadata)
    output['CORE'] = float(_core.groups()[0]) if _core else 0.0
    _maxit = re.search('MAXIT'+pattern, metadata)
    output['MAXIT'] = int(_maxit.groups()[0]) if _maxit else 25
    _thr = re.search('THR'+pattern, metadata)
    output['THR'] = float(_thr.groups()[0]) if _thr else 1e-5
    _thrres = re.search('THRRES'+pattern, metadata)
    output['THRRES'] = float(_thrres.groups()[0]) if _thrres else 0.1
    _nroot = re.search('NROOT'+pattern, metadata)
    output['NROOT'] = int(_nroot.groups()[0]) if _nroot else 1

    # If the FCIDump file resulted from an unrestricted spin calculation the indices will label spin
    # rather than molecular orbitals. This means, that a line must exist which encodes the
    # coefficient for the spin orbital with index (norb*2, norb*2). By checking for such a line we
    # can distinguish between unrestricted and restricted FCIDump files.
    _uhf = bool(re.search(r'.*(\s+{}\s+{}\s+0\s+0)'.format(norb*2, norb*2),
                          fcidump_str[namelist_end.start(0):]))

    # the rest of the FCIDump will hold lines of the form x i a j b
    # a few cases have to be treated differently:
    # i, a, j and b are all zero: x is the core energy
    # TODO: a, j and b are all zero: x is the energy of the i-th MO  (often not supported)
    # j and b are both zero: x is the 1e-integral between i and a (x = <i|h|a>)
    # otherwise: x is the Coulomb integral ( x = (ia|jb) )
    hij = np.zeros((norb, norb))
    hij_elements = set(itertools.product(range(norb), repeat=2))
    hijkl = np.zeros((norb, norb, norb, norb))
    hijkl_elements = set(itertools.product(range(norb), repeat=4))
    hij_b = hijkl_ab = hijkl_ba = hijkl_bb = None
    hij_b_elements = hijkl_ab_elements = hijkl_ba_elements = hijkl_bb_elements = set()
    if _uhf:
        beta_range = [n+norb for n in range(norb)]
        hij_b = np.zeros((norb, norb))
        hij_b_elements = set(itertools.product(beta_range, repeat=2))
        hijkl_ab = np.zeros((norb, norb, norb, norb))
        hijkl_ba = np.zeros((norb, norb, norb, norb))
        hijkl_bb = np.zeros((norb, norb, norb, norb))
        hijkl_ab_elements = set(itertools.product(
            range(norb), range(norb), beta_range, beta_range
        ))
        hijkl_ba_elements = set(itertools.product(
            beta_range, beta_range, range(norb), range(norb)
        ))
        hijkl_bb_elements = set(itertools.product(beta_range, repeat=4))
    orbital_data = fcidump_str[namelist_end.end(0):].split('\n')
    for orbital in orbital_data:
        if not orbital:
            continue
        x = float(orbital.split()[0])
        # Note: differing naming than ijkl due to E741 and this iajb is inline with this:
        # https://hande.readthedocs.io/en/latest/manual/integrals.html#fcidump-format
        i, a, j, b = [int(i) for i in orbital.split()[1:]]  # pylint: disable=invalid-name
        if i == a == j == b == 0:
            output['ecore'] = x
        elif a == j == b == 0:
            # TODO: x is the energy of the i-th MO
            continue
        elif j == b == 0:
            try:
                hij_elements.remove((i-1, a-1))
                hij[i-1][a-1] = x
            except KeyError as ex:
                if _uhf:
                    hij_b_elements.remove((i-1, a-1))
                    hij_b[i-1-norb][a-1-norb] = x
                else:
                    raise QiskitChemistryError("Unkown 1-electron integral indices encountered in \
                            '{}'".format((i, a))) from ex
        else:
            try:
                hijkl_elements.remove((i-1, a-1, j-1, b-1))
                hijkl[i-1][a-1][j-1][b-1] = x
            except KeyError as ex:
                if _uhf:
                    try:
                        hijkl_ab_elements.remove((i-1, a-1, j-1, b-1))
                        hijkl_ab[i-1][a-1][j-1-norb][b-1-norb] = x
                    except KeyError:
                        try:
                            hijkl_ba_elements.remove((i-1, a-1, j-1, b-1))
                            hijkl_ba[i-1-norb][a-1-norb][j-1][b-1] = x
                        except KeyError:
                            hijkl_bb_elements.remove((i-1, a-1, j-1, b-1))
                            hijkl_bb[i-1-norb][a-1-norb][j-1-norb][b-1-norb] = x
                else:
                    raise QiskitChemistryError("Unkown 2-electron integral indices encountered in \
                            '{}'".format((i, a, j, b))) from ex

    # iterate over still empty elements in 1-electron matrix and populate with symmetric ones
    # if any elements are not populated these will be zero
    _permute_1e_ints(hij, hij_elements, norb)

    if _uhf:
        # do the same for beta spin
        _permute_1e_ints(hij_b, hij_b_elements, norb, beta=True)

    # do the same of the 2-electron 4D matrix
    _permute_2e_ints(hijkl, hijkl_elements, norb)

    if _uhf:
        # do the same for beta spin
        _permute_2e_ints(hijkl_bb, hijkl_bb_elements, norb, beta=2)
        _permute_2e_ints(hijkl_ab, hijkl_ab_elements, norb, beta=1)  # type: ignore
        _permute_2e_ints(hijkl_ba, hijkl_ba_elements, norb, beta=1)  # type: ignore

        # assert that EITHER hijkl_ab OR hijkl_ba were given
        if np.allclose(hijkl_ab, 0.0) == np.allclose(hijkl_ba, 0.0):
            raise QiskitChemistryError("Encountered mixed sets of indices for the 2-electron \
                    integrals. Either alpha/beta or beta/alpha matrix should be specified.")

        if np.allclose(hijkl_ba, 0.0):
            hijkl_ba = hijkl_ab.transpose()

    output['hij'] = hij
    output['hij_b'] = hij_b
    output['hijkl'] = hijkl
    output['hijkl_ba'] = hijkl_ba
    output['hijkl_bb'] = hijkl_bb

    return output


def _permute_1e_ints(hij: np.ndarray,
                     elements: Set[Tuple[int, ...]],
                     norb: int,
                     beta: bool = False) -> None:
    for elem in elements.copy():
        shifted = tuple(e-(beta * norb) for e in elem)
        hij[shifted] = hij[shifted[::-1]]
        elements.remove(elem)


def _permute_2e_ints(hijkl: np.ndarray,
                     elements: Set[Tuple[int, ...]],
                     norb: int,
                     beta: int = 0) -> None:
    # pylint: disable=wrong-spelling-in-comment
    for elem in elements.copy():
        shifted = tuple(e-((e >= norb) * norb) for e in elem)
        # initially look for "transposed" element if spins are equal
        if beta != 1 and elem[::-1] not in elements:
            hijkl[shifted] = hijkl[shifted[::-1]]
            elements.remove(elem)
            continue
        # then look at permutations of indices within the bra and ket respectively
        bra_perms = set(itertools.permutations(elem[:2]))
        ket_perms = set(itertools.permutations(elem[2:]))
        if beta == 1:
            # generally (ij|ab) != (ab|ij)
            # thus, the possible permutations are much less when the spins differ
            permutations = itertools.product(bra_perms, ket_perms)
        else:
            # ( ij | kl ) gives { ( ij | kl ), ( ij | lk ), ( ji | kl ), ( ji | lk ) }
            # AND { ( kl | ij ), ( kl | ji ), ( lk | ij ), ( lk | ji ) }
            # BUT NOT ( ik | jl ) etc.
            permutations = itertools.chain(
                itertools.product(bra_perms, ket_perms),
                itertools.product(ket_perms, bra_perms)
            )
        for perm in {e1 + e2 for e1, e2 in permutations}:
            if perm in elements:
                continue
            hijkl[shifted] = hijkl[tuple([e-((e >= norb) * norb) for e in perm])]
            elements.remove(elem)
            break
