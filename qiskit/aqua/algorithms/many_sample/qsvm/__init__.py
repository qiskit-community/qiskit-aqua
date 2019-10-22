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

""" Quantum SVM packages """

from ._qsvm_abc import _QSVM_ABC
from ._qsvm_binary import _QSVM_Binary
from ._qsvm_multiclass import _QSVM_Multiclass
from ._qsvm_estimator import _QSVM_Estimator

__all__ = ['_QSVM_ABC',
           '_QSVM_Binary',
           '_QSVM_Multiclass',
           '_QSVM_Estimator'
           ]
