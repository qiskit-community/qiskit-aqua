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

from .qsvm_kernel_abc import QSVM_Kernel_ABC
from .qsvm_kernel_binary import QSVM_Kernel_Binary
from .qsvm_kernel_multiclass import QSVM_Kernel_Multiclass
from .qsvm_kernel_estimator import QSVM_Kernel_Estimator

__all__ = ['QSVM_Kernel_ABC',
           'QSVM_Kernel_Binary',
           'QSVM_Kernel_Multiclass',
           'QSVM_Kernel_Estimator'
           ]
