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

import numpy as np


def tensorproduct(*args):
    """
    Calculate tensor product.

    m = tensorproduct(a,b,c,...) returns the kronecker product of its arguments.
    Each argument should either be a tensor, or a tuple containing a
    tensor and an integer, and tensor is put in zero-index slot.
    In the latter case, the integer specifies the repeat count for the tensor,
    e.g. tensorproduct(a,(b,3),c) = tensorproduct(a,b,b,b,c).

    Args:
            - args:
    Returns:
            np.ndarray: the tensor product
    """
    M = 1
    for j in range(len(args)):
        if isinstance(args[j], tuple):
            m = args[j][0] if isinstance(args[j][0], np.ndarray) else np.asarray(args[j][0])
            for k in range(args[j][1]):
                M = np.kron(M, m)
        else:
            m = args[j] if isinstance(args[j], np.ndarray) else np.asarray(args[j])
            M = np.kron(M, m)
    return M
