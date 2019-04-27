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

from ._qsvm_abc import _QSVM_ABC
from ._qsvm_binary import _QSVM_Binary
from ._qsvm_multiclass import _QSVM_Multiclass
from ._qsvm_estimator import _QSVM_Estimator

__all__ = ['_QSVM_ABC',
           '_QSVM_Binary',
           '_QSVM_Multiclass',
           '_QSVM_Estimator'
           ]
