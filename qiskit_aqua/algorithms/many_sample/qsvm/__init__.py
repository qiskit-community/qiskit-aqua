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

from .svm_qkernel_abc import SVM_QKernel_ABC
from .svm_qkernel_binary import SVM_QKernel_Binary
from .svm_qkernel_multiclass import SVM_QKernel_Multiclass
from .qkernel_svm_estimator import QKernalSVM_Estimator

__all__ = ['SVM_QKernel_ABC',
           'SVM_QKernel_Binary',
           'SVM_QKernel_Multiclass',
           'QKernalSVM_Estimator'
           ]
