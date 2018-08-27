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

import logging

import numpy as np
from cvxopt import matrix, solvers

logger = logging.getLogger(__name__)


def optimize_SVM(K, y, scaling=None, max_iters=500, show_progress=False):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    H = np.outer(y, y) * K
    f = -np.ones(y.shape)
    if scaling is None:
        scaling = np.sum(np.sqrt(f * f))
    f /= scaling

    tolerance = 1e-2
    n = K.shape[1]

    P = matrix(H)
    q = matrix(f)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(y, y.T.shape)
    b = matrix(np.zeros(1), (1, 1))
    solvers.options['maxiters'] = max_iters
    solvers.options['show_progress'] = show_progress

    ret = solvers.qp(P, q, G, h, A, b, kktsolver='ldl')
    alpha = np.asarray(ret['x']) * scaling
    avg_y = np.sum(y)
    avg_mat = (alpha * y).T.dot(K.dot(np.ones(y.shape)))
    b = (avg_y - avg_mat) / n

    support = alpha > tolerance
    logger.debug('Solving QP problem is completed.')
    return alpha.flatten(), b.flatten(), support.flatten()
