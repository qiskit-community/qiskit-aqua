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

from ._qsvm_kernel_abc import _QSVM_Kernel_ABC
from ._qsvm_kernel_binary import _QSVM_Kernel_Binary
from ._qsvm_kernel_multiclass import _QSVM_Kernel_Multiclass
from ._qsvm_kernel_estimator import _QSVM_Kernel_Estimator

__all__ = ['_QSVM_Kernel_ABC',
           '_QSVM_Kernel_Binary',
           '_QSVM_Kernel_Multiclass',
           '_QSVM_Kernel_Estimator'
           ]
