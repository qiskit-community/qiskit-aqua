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

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def gaussian(training_size, test_size, n, PLOT_DATA):
    sigma = 1
    if n == 2:
        class_labels = [r'A', r'B']
        label_train = np.zeros(2*(training_size+test_size))
        sample_train = []
        sampleA = [[0 for x in range(n)] for y in range(training_size+test_size)]
        sampleB = [[0 for x in range(n)] for y in range(training_size+test_size)]
        randomized_vector1 = np.random.randint(2, size=n)
        randomized_vector2 = (randomized_vector1+1) % 2
        for tr in range(training_size+test_size):
            for feat in range(n):
                if randomized_vector1[feat] == 0:
                    sampleA[tr][feat] = np.random.normal(-1/2, sigma, None)
                elif randomized_vector1[feat] == 1:
                    sampleA[tr][feat] = np.random.normal(1/2, sigma, None)

                if randomized_vector2[feat] == 0:
                    sampleB[tr][feat] = np.random.normal(-1/2, sigma, None)
                elif randomized_vector2[feat] == 1:
                    sampleB[tr][feat] = np.random.normal(1/2, sigma, None)

        sample_train = [sampleA, sampleB]
        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}

        if PLOT_DATA:
            if not HAS_MATPLOTLIB:
                raise NameError('Matplotlib not installed. Plase install it before plotting')
            # fig1 = plt.figure()
            for k in range(0, 2):
                plt.scatter(sample_train[label_train == k, 0][:training_size],
                            sample_train[label_train == k, 1][:training_size])

            plt.title("Gaussians")
            plt.show()

        return sample_train, training_input, test_input, class_labels
    elif n == 3:
        class_labels = [r'A', r'B', r'C']
        label_train = np.zeros(3*(training_size+test_size))
        sample_train = []
        sampleA = [[0 for x in range(n)] for y in range(training_size+test_size)]
        sampleB = [[0 for x in range(n)] for y in range(training_size+test_size)]
        sampleC = [[0 for x in range(n)] for y in range(training_size+test_size)]
        randomized_vector1 = np.random.randint(3, size=n)
        randomized_vector2 = (randomized_vector1+1) % 3
        randomized_vector3 = (randomized_vector2+1) % 3
        for tr in range(training_size+test_size):
            for feat in range(n):
                if randomized_vector1[feat] == 0:
                    sampleA[tr][feat] = np.random.normal(2*1*np.pi/6, sigma, None)
                elif randomized_vector1[feat] == 1:
                    sampleA[tr][feat] = np.random.normal(2*3*np.pi/6, sigma, None)
                elif randomized_vector1[feat] == 2:
                    sampleA[tr][feat] = np.random.normal(2*5*np.pi/6, sigma, None)

                if randomized_vector2[feat] == 0:
                    sampleB[tr][feat] = np.random.normal(2*1*np.pi/6, sigma, None)
                elif randomized_vector2[feat] == 1:
                    sampleB[tr][feat] = np.random.normal(2*3*np.pi/6, sigma, None)
                elif randomized_vector2[feat] == 2:
                    sampleB[tr][feat] = np.random.normal(2*5*np.pi/6, sigma, None)

                if randomized_vector3[feat] == 0:
                    sampleC[tr][feat] = np.random.normal(2*1*np.pi/6, sigma, None)
                elif randomized_vector3[feat] == 1:
                    sampleC[tr][feat] = np.random.normal(2*3*np.pi/6, sigma, None)
                elif randomized_vector3[feat] == 2:
                    sampleC[tr][feat] = np.random.normal(2*5*np.pi/6, sigma, None)

        sample_train = [sampleA, sampleB, sampleC]
        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+training_size+test_size+lindex] = 2
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (3*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}

        if PLOT_DATA:
            if not HAS_MATPLOTLIB:
                raise NameError('Matplotlib not installed. Plase install it before plotting')
            # fig1 = plt.figure()
            for k in range(0, 3):
                plt.scatter(sample_train[label_train == k, 0][:training_size],
                            sample_train[label_train == k, 1][:training_size])

            plt.title("Gaussians")
            plt.show()

        return sample_train, training_input, test_input, class_labels
    else:
        raise ValueError("Gaussian presently only supports 2 or 3 qubits")
