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
# =============================================================================
"""
For coloring Pauli Graph for transforming paulis into grouped Paulis
"""

import copy

import numpy as np


class PauliGraph(object):
    """
        Pauli Graph
    """

    def __init__(self, paulis, mode="largest-degree"):
        self.nodes, self.weights = self._create_nodes(paulis)  # must be pauli list
        self._nqbits = self._get_nqbits()
        self.edges = self._create_edges()
        self._grouped_paulis = self._coloring(mode)

    def _create_nodes(self, paulis):
        """
        Check the validity of the pauli list and return immutable list as list of nodes

        Args:
            paulis: list of [weight, Pauli object]

        Returns:
            Pauli object as immutable list
        """
        pauliOperators = [x[1] for x in paulis]
        pauliWeights = [x[0] for x in paulis]
        return tuple(pauliOperators), tuple(pauliWeights)  # fix their ordering

    def _get_nqbits(self):
        nqbits = self.nodes[0].numberofqubits
        for i in range(1, len(self.nodes)):
            assert nqbits == self.nodes[i].numberofqubits, "different number of qubits"
        return nqbits

    def _create_edges(self):
        """
        create edges (i,j) if i and j is not commutable under Paulis

        Returns:
            dictionary of graph connectivity with node index as key and list of neighbor as values
        """
        #assert self.nodes is not None, "No nodes found."
        edges = {i: [] for i in range(len(self.nodes))}
        for i in range(len(self.nodes)):
            nodei = self.nodes[i]
            for j in range(i+1, len(self.nodes)):
                nodej = self.nodes[j]
                isCommutable = True
                for k in range(self._nqbits):
                    if ((nodei.v[k] == 0 and nodei.w[k] == 0) or
                        (nodej.v[k] == 0 and nodej.w[k] == 0) or
                            (nodei.v[k] == nodej.v[k] and nodei.w[k] == nodej.w[k])):
                        pass
                    else:
                        isCommutable = False
                        break
                if not isCommutable:
                    edges[i].append(j)
                    edges[j].append(i)
        return edges

    def _coloring(self, mode="largest-degree"):
        if mode == "largest-degree":
            degree = [(i, len(self.edges[i])) for i in range(len(self.nodes))]  # list of (nodeid, degree)
            degree.sort(key=lambda x: x[1], reverse=True)  # sorted by largest degree
            # -1 means not colored, 0 ... len(self.nodes)-1 is valid colored
            color = [-1 for i in range(len(self.nodes))]
            # coloring start
            for nodei, _ in degree:
                ineighbors = self.edges[nodei]  # get neighbors of nodei
                ineighborsColors = set([color[j] for j in ineighbors if color[j] >= 0])  # get colors of ineighbors
                for c in range(len(self.nodes)):
                    if c not in ineighborsColors:
                        color[nodei] = c
                        break
            # coloring end
            assert (np.array(color) >= 0).all(), "Uncolored node exists!"

            # post-processing to grouped_paulis
            maxColor = np.max(color)  # the color used is 0, 1, 2, ..., maxColor
            temp_gp = []  # list of indices of grouped paulis
            for c in range(maxColor+1):  # maxColor is included
                temp_gp.append([i for i, icolor in enumerate(color) if icolor == c])

            # create _grouped_paulis as dictated in the operator.py
            gp = []
            for c in range(maxColor+1):  # maxColor is included
                # add all paulis
                gp.append([[self.weights[i], self.nodes[i]] for i in temp_gp[c]])

            # create header (measurement basis)
            for i, groupi in enumerate(gp):
                header = [0.0, copy.deepcopy(groupi[0][1])]
                for _, p in groupi:
                    for k in range(self._nqbits):
                        if p.v[k] == 1 or p.w[k] == 1:
                            header[1].v[k] = p.v[k]
                            header[1].w[k] = p.w[k]
                gp[i].insert(0, header)
            return gp
        else:
            return self._coloring("largest-degree")  # this is the default implementation

    @property
    def grouped_paulis(self):
        """
        Getter of grouped Pauli list.
        """
        return self._grouped_paulis
