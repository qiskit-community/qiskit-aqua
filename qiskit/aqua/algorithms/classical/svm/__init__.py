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

""" svm packages """

from ._svm_classical_abc import _SVM_Classical_ABC
from ._svm_classical_binary import _SVM_Classical_Binary
from ._svm_classical_multiclass import _SVM_Classical_Multiclass
from ._rbf_svc_estimator import _RBF_SVC_Estimator

__all__ = [
    '_SVM_Classical_ABC',
    '_SVM_Classical_Binary',
    '_SVM_Classical_Multiclass',
    '_RBF_SVC_Estimator',
]
