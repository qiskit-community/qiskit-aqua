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

""" common module """

from collections import OrderedDict

import numpy as np

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import StateFn


def random_graph(n, weight_range=10, edge_prob=0.3, negative_weight=True,
                 savefile=None, seed=None):
    """Generate random Erdos-Renyi graph.

    Args:
        n (int): number of nodes.
        weight_range (int): weights will be smaller than this value,
            in absolute value. range: [1, weight_range).
        edge_prob (float): probability of edge appearing.
        negative_weight (bool): allow to have edge with negative weights
        savefile (str or None): name of file where to save graph.
        seed (int or None): random seed - if None, will not initialize.

    Returns:
        numpy.ndarray: adjacency matrix (with weights).

    """
    assert weight_range >= 0
    if seed:
        aqua_globals.random_seed = seed
    w = np.zeros((n, n))
    m = 0
    for i in range(n):
        for j in range(i + 1, n):
            if aqua_globals.random.random() <= edge_prob:
                w[i, j] = aqua_globals.random.integers(1, weight_range)
                if aqua_globals.random.random() >= 0.5 and negative_weight:
                    w[i, j] *= -1
                m += 1
    w += w.T
    if savefile:
        with open(savefile, 'w') as outfile:
            outfile.write('{} {}\n'.format(n, m))
            for i in range(n):
                for j in range(i + 1, n):
                    if w[i, j] != 0:
                        outfile.write('{} {} {}\n'.format(i + 1, j + 1, w[i, j]))
    return w


def random_number_list(n, weight_range=100, savefile=None, seed=None):
    """Generate a set of positive integers within the given range.

    Args:
        n (int): size of the set of numbers.
        weight_range (int): maximum absolute value of the numbers.
        savefile (str or None): write numbers to this file.
        seed (Union(int,None)): random seed - if None, will not initialize.

    Returns:
        numpy.ndarray: the list of integer numbers.
    """
    if seed:
        aqua_globals.random_seed = seed

    number_list = aqua_globals.random.integers(low=1, high=(weight_range + 1), size=n)
    if savefile:
        with open(savefile, 'w') as outfile:
            for i in range(n):
                outfile.write('{}\n'.format(number_list[i]))
    return number_list


def read_numbers_from_file(filename):
    """Read numbers from a file

    Args:
        filename (str): name of the file.

    Returns:
        numpy.ndarray: list of numbers as a numpy.ndarray.
    """
    numbers = []
    with open(filename) as infile:
        for line in infile:
            assert int(round(float(line))) == float(line)
            numbers.append(int(round(float(line))))
    return np.array(numbers)


def parse_gset_format(filename):
    """Read graph in Gset format from file.

    Args:
        filename (str): name of the file.

    Returns:
        numpy.ndarray: adjacency matrix as a 2D numpy array.
    """
    n = -1
    with open(filename) as infile:
        header = True
        m = -1
        count = 0
        for line in infile:
            v = map(lambda e: int(e), line.split())  # pylint: disable=unnecessary-lambda
            if header:
                n, m = v
                w = np.zeros((n, n))
                header = False
            else:
                s__, t__, _ = v
                s__ -= 1  # adjust 1-index
                t__ -= 1  # ditto
                w[s__, t__] = t__
                count += 1
        assert m == count
    w += w.T
    return w


def get_gset_result(x):
    """Get graph solution in Gset format from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        Dict[int, int]: graph solution in Gset format.
    """
    return {i + 1: 1 - x[i] for i in range(len(x))}


def sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector (numpy.ndarray or dict): state vector or counts.

    Returns:
        numpy.ndarray: binary string as numpy.ndarray of ints.
    """
    if isinstance(state_vector, (OrderedDict, dict)):
        # get the binary string with the largest count
        binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    elif isinstance(state_vector, StateFn):
        binary_string = list(state_vector.sample().keys())[0]
        x = np.asarray([int(y) for y in reversed(list(binary_string))])
        return x
    else:
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x
