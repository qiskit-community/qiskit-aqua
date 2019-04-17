# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
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
"""
Maximum Likelihood post-processing for the Amplitude Estimation algorithm.
"""

import logging
import numpy as np
from scipy.optimize import bisect

from qiskit.aqua import AquaError
from .mle_utils import loglik, bisect_max
from .ci_utils import (chi2_quantile, normal_quantile, fisher_information,
                       d_logprob)

logger = logging.getLogger(__name__)


class MaximumLikelihood:
    """
    Maximum Likelihood post-processing for the Amplitude Estimation algorithm.
    """

    def __init__(self, ae):
        """
        @brief Initialise with AmplitudeEstimation instance and compute the
               number of shots
        @param ae An instance of AmplitudeEstimation
        """
        self.ae = ae

        # Find number of shots
        if "counts" in self.ae._ret.keys():  # qasm_simulator
            self._shots = sum(self.ae._ret['counts'].values())
        else:  # statevector_simulator
            self._shots = 1

    def loglik_wrapper(self, theta):
        """
        @brief Wrapper for the loglikelihood, measured values, probabilities
               and number of shots already put in and only dependent on the
               exact value `a`, called `theta` now.
        @param theta The exact value
        @return The likelihood of the AE measurements, if `theta` were the
                exact value
        """
        return loglik(theta,
                      self.ae._m,
                      np.asarray(self.ae._ret['values']),
                      np.asarray(self.ae._ret['probabilities']),
                      self._shots)

    def mle(self):
        """
        @brief Compute the Maximum Likelihood Estimator (MLE)
        @return The MLE for the previous AE run
        @note Before calling this method, call the method `run` of the
              AmplitudeEstimation instance
        """
        self._qae = self.ae._ret['estimation']

        # Compute the two in which we the look for values above the
        # threshold
        M = 2**self.ae._m
        y = M * np.arcsin(np.sqrt(self._qae)) / np.pi
        left_gridpoint = np.sin(np.pi * (y - 1) / M)**2
        right_gridpoint = np.sin(np.pi * (y + 1) / M)**2
        bubbles = [left_gridpoint, self._qae, right_gridpoint]

        # Find global maximum amongst the local maxima, which are
        # located in between the drops
        a_opt = self._qae
        loglik_opt = self.loglik_wrapper(a_opt)
        for a, b in zip(bubbles[:-1], bubbles[1:]):
            locmax, val = bisect_max(self.loglik_wrapper, a, b, retval=True)
            if val > loglik_opt:
                a_opt = locmax
                loglik_opt = val

        # Convert the value to an estimation
        val_opt = self.ae.a_factory.value_to_estimation(a_opt)

        self._mle = a_opt
        self._mapped_mle = val_opt

        return val_opt

    def ci(self, alpha, kind="likelihood_ratio"):
        """
        @brief Compute the (1 - alpha) confidence interval (CI) with the method
               specified in `kind`
        @param alpha Confidence level: asymptotically 100(1 - alpha)% of the
                     data will be contained in the CI
        @return The confidence interval
        """

        if kind == "fisher":
            # Compute the predicted standard deviation
            std = np.sqrt(self._shots * fisher_information(self._mle, self.ae._m))

            # Set up the (1 - alpha) symmetric confidence interval
            ci = self._mle + normal_quantile(alpha) / std * np.array([-1, 1])

        elif kind == "observed_fisher":
            ai = np.asarray(self.ae._ret['values'])
            pi = np.asarray(self.ae._ret['probabilities'])

            # Calculate the observed Fisher information
            observed_information = np.sum(self._shots * pi * d_logprob(ai, self._mle, self.ae._m)**2)

            # Set up the (1 - alpha) symmetric confidence interval
            std = np.sqrt(observed_information)
            ci = self._mle + normal_quantile(alpha) / std * np.array([-1, 1])

        elif kind == "likelihood_ratio":
            # Compute the two in which we the look for values above the
            # threshold
            M = 2**self.ae._m
            y = M * np.arcsin(np.sqrt(self._qae)) / np.pi
            left_gridpoint = np.sin(np.pi * (y - 1) / M)**2
            right_gridpoint = np.sin(np.pi * (y + 1) / M)**2
            bubbles = [left_gridpoint, self._qae, right_gridpoint]

            # The threshold above which the likelihoods are in the
            # confidence interval
            loglik_ref = self.loglik_wrapper(self._mle)
            thres = loglik_ref - chi2_quantile(alpha) / 2

            # Store the boundaries of the confidence interval
            lower = upper = self._mle

            # Iterate over the bubbles in between the drops, check if they
            # surpass the threshold and if yes add the part that does to the
            # confidence interval
            for a, b in zip(bubbles[:-1], bubbles[1:]):
                locmax, val = bisect_max(self.loglik_wrapper, a, b, retval=True)
                if val >= thres:
                    left = bisect(lambda x: self.loglik_wrapper(x) - thres, a, locmax)
                    right = bisect(lambda x: self.loglik_wrapper(x) - thres, locmax, b)
                    lower = np.minimum(lower, left)
                    upper = np.maximum(upper, right)

            ci = np.append(lower, upper)
        else:
            raise AquaError("Confidence interval kind {} not implemented.".format(kind))

        return ci
