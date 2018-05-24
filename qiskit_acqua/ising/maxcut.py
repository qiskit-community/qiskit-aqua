# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

# Convert maxcut instances into Pauli list
# Deal with Gset format. See https://web.stanford.edu/~yyye/yyye/Gset/
# Design the maxcut object `w` as a two-dimensional np.array
# e.g., w[i, j] = x means that the weight of a edge between i and j is x
# Note that the weights are symmetric, i.e., w[j, i] = x always holds.

import logging
import random
from typing import Tuple, Dict, List

import numpy as np

from qiskit_acqua import Operator

logger = logging.getLogger(__name__)


def random_maxcut(n, savefile=None, seed=None):
    if seed:
        random.seed(seed)
    w = {}
    m = 0
    for j in range(n):
        for i in range(j):
            if random.random() >= 0.5:
                w[i, j] = w[j, i] = 1
                m += 1

    if savefile:
        with open(savefile, 'w') as outfile:
            outfile.write('{} {}\n'.format(n, m))
            for (i, j), t in sorted(w.items()):
                if i < j:
                    outfile.write('{} {} {}\n'.format(i + 1, j + 1, t))
    return convert_maxcut_to_qubitops(n, w)


def convert_maxcut_to_qubitops(n: int, w: Dict[Tuple[int, int], int]) -> \
        Tuple[Operator, float, Dict[Tuple[int, int], int], int]:
    pauli_list = []
    cost_shift = 0
    for (i, j), t in w.items():
        if i < j:
            cost_shift += t
            label = list('I' * n)
            label[i] = label[j] = 'Z'
            pauli_list.append({'coeff': {'imag': 0.0, 'real': t}, 'label': ''.join(label)})
    return Operator.load_from_dict({'paulis': pauli_list}), cost_shift, w, n


def parse_gset_format(filename: str):
    n = -1
    with open(filename) as infile:
        header = True
        w: Dict[Tuple[int, int], int] = None
        m = -1
        count = 0
        for line in infile:
            v = map(lambda e: int(e), line.split())
            if header:
                n, m = v
                w = {}
                header = False
            else:
                s, t, x = v
                s -= 1  # adjust 1-index
                t -= 1  # ditto
                w[s, t] = w[t, s] = x
                count += 1
        assert m == count
    return convert_maxcut_to_qubitops(n, w)


def maxcut_value(x, w):
    total = 0
    for (i, j), t in w.items():
        total += t * x[i] * (1.0 - x[j])
    return total


def maxcut_obj(result, offset):
    return -result['energy'] * 0.5 + offset * 0.5


def convert_eigevecs(n: int, result: List[float]) -> Dict[int, int]:
    v = np.array(result)
    k = np.argmax(v)
    x = {}
    for i in range(n):
        x[i] = k % 2
        k >>= 1
    return x


def convert_gset_result(x: Dict) -> Dict[int, int]:
    return {k + 1: v for k, v in x.items()}
