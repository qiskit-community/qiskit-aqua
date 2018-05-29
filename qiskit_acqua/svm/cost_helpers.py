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

from collections import Counter

import numpy as np


def assign_label(key, class_labels):
    # If odd number of qubits and two labels we use majority vote
    if len(class_labels) == 2 and int(len(list(key)) % 2) != 0:
        vote_count = Counter(key)
        top_count = vote_count.most_common(2)
        result = int(top_count[0][0])
        return class_labels[result]
    # If even number of qubits and two labels we use parity
    elif len(class_labels) == 2:
        hamming_weight = sum([int(k) for k in list(key)])
        is_odd_parity = hamming_weight & 1

        if is_odd_parity:
            return class_labels[1]
        else:
            return class_labels[0]
    # If three labels we use part-parity {ex. for 2 qubits: [00], [01,10], [11] would be the three labels}
    elif len(class_labels) == 3:
        first_half = int(np.floor(len(list(key))/2))
        modulo = int(len(list(key)) % 2)
        hamming_weight_1 = sum([int(k) for k in list(key)[0:first_half+modulo]])  # First half of key
        hamming_weight_2 = sum([int(k) for k in list(key)[first_half+modulo:]])   # Second half of key
        is_odd_parity_1 = hamming_weight_1 & 1
        is_odd_parity_2 = hamming_weight_2 & 1

        if (not is_odd_parity_1) & (not is_odd_parity_2):  # Both halves even
            return class_labels[0]
        elif is_odd_parity_1 & is_odd_parity_2:  # Both halves odd
            return class_labels[2]
        else:  # One half even, one half odd
            return class_labels[1]
    else:
        total_size = 2**len(list(key))
        class_step = np.floor(total_size/len(class_labels))
        key_order = int(np.floor(int(key, 2)/class_step))
        if key_order < len(class_labels):
            return class_labels[key_order]
        else:
            return class_labels[-1]


def cost_estimate_sigmoid(shots, probs, expected_category):

    p = probs.get(expected_category)

    if p < 0:
        p = 0
    elif p > 1:
        p = 1

    probs_without_measured_expectation = [v for key, v in probs.items() if key != expected_category]

    number_of_classes = len(probs)

    sig = None
    if number_of_classes == 2:
        if np.isclose(p, 0.0):
            sig = 1
        elif np.isclose(p, 1.0):
            sig = 0
        else:
            denominator = np.sqrt(2*p*(1-p))
            x = np.sqrt(shots)*(0.5-p)/denominator
            sig = 1/(1+np.exp(-x))
    elif number_of_classes == 3:
        if np.isclose(p, 0.0):
            sig = 1
        elif np.isclose(p, 1.0):
            sig = 0
        else:
            denominator = np.sqrt(2*p*(1-p))
            numerator = np.sqrt(shots)*((1+abs(probs_without_measured_expectation[0]-probs_without_measured_expectation[1]))/3-p)
            x = numerator/denominator
            sig = 1/(1+np.exp(-x))
    return sig


def return_probabilities(counts, class_labels):
    hits = sum(counts.values())

    result = {class_labels[p]: 0 for p in range(len(class_labels))}
    for (key, item) in counts.items():
        # The different measurement transforms into a class result happens in assign_label
        hw = assign_label(key, class_labels)
        result[hw] += counts[key]/hits

    return result
