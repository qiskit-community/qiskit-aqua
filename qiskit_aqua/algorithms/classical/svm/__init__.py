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

from .svm_classical_abc import SVM_Classical_ABC
from .svm_classical_binary import SVM_Classical_Binary
from .svm_classical_multiclass import SVM_Classical_Multiclass
from .rbf_svc_estimator import RBF_SVC_Estimator

__all__ = ['SVM_Classical_ABC',
           'SVM_Classical_Binary',
           'SVM_Classical_Multiclass'
           'RBF_SVC_Estimator']
