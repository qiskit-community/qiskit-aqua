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

# the APIs (get_points_and_labels and get_points) are just helpers
import numpy as np


def get_points_and_labels(input, class_labels):
    first_array = input[class_labels[0]]
    second_array = input[class_labels[1]]
    total_array = np.concatenate([first_array, second_array])

    test_label_0 = np.ones(len(first_array))
    test_label_1 = -1*np.ones(len(second_array))
    test_label = np.concatenate((test_label_0, test_label_1))

    label_to_class = {1: class_labels[0], -1: class_labels[1]}

    return total_array, test_label, label_to_class


def get_points(input, class_labels):
    first_array = input[class_labels[0]]
    second_array = input[class_labels[1]]
    total_array = np.concatenate([first_array, second_array])

    label_to_class = {1: class_labels[0], -1: class_labels[1]}

    return total_array, label_to_class
