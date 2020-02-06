# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" tensor product """

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
    m_l = 1
    for j, _ in enumerate(args):
        if isinstance(args[j], tuple):
            m = args[j][0] if isinstance(args[j][0], np.ndarray) else np.asarray(args[j][0])
            for _ in range(args[j][1]):
                m_l = np.kron(m_l, m)
        else:
            m = args[j] if isinstance(args[j], np.ndarray) else np.asarray(args[j])
            m_l = np.kron(m_l, m)
    return m_l
