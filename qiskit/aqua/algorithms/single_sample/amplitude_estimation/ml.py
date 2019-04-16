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
"""
The Amplitude Estimation Algorithm.
"""

import logging
import numpy as np

from qiskit.aqua import AquaError
from .mle_utils import loglik, bisect_max
from .ci_utils import (chi2_quantile, normal_quantile, fisher_information,
                       d_logprob)

logger = logging.getLogger(__name__)


class MaximumLikelihood:
    def __init__(self, ae):
        self.ae = ae

        # Find number of shots
        if "counts" in self.ae._ret.keys():  # qasm_simulator
            self._shots = sum(self.ae._ret['counts'].values())
        else:  # statevector_simulator
            self._shots = 1

        # Result dictionary
        self._ret = {}

    def loglik_wrapper(self, theta):
        """
        Wrapper for the loglikelihood, measured values, probabilities
        and number of shots already put in and only dependent on the
        exact value `a`, called `theta` now.
        """
        return loglik(theta,
                      self.ae._m,
                      np.asarray(self.ae._ret['values']),
                      np.asarray(self.ae._ret['probabilities']),
                      self._shots)

    def mle(self):
        # Compute the singularities of the log likelihood (= QAE grid points)
        drops = np.sin(np.pi * np.linspace(0, 0.5,
                                           num=int(self.ae._m / 2),
                                           endpoint=False))**2

        drops = np.append(drops, 1)  # 1 is also a singularity

        # Find global maximum amongst the local maxima, which are
        # located in between the drops
        a_opt = self.ae._ret['estimation']
        loglik_opt = self.loglik_wrapper(a_opt)
        for a, b in zip(drops[:-1], drops[1:]):
            local, loglik_local = bisect_max(self.loglik_wrapper, a, b, retval=True)
            if loglik_local > loglik_opt:
                a_opt = local
                loglik_opt = loglik_local

        # Convert the value to an estimation
        self._ret['mle'] = self.ae.a_factory.value_to_estimation(a_opt)
        self._ret['mle_value'] = a_opt

        return self._ret

    def ci(self, alpha, kind="likelihood_ratio"):

        mle = self._ret['mle_value']

        if kind == "fisher":
            std = np.sqrt(self._shots * fisher_information(mle, self.ae._m))
            ci = mle + normal_quantile(alpha) / std * np.array([-1, 1])

        elif kind == "observed_fisher":
            ai = np.asarray(self.ae._ret['values'])
            pi = np.asarray(self.ae._ret['probabilities'])
            observed_information = np.sum(self._shots * pi * d_logprob(ai, mle, self.ae._m)**2)
            std = np.sqrt(observed_information)
            ci = mle + normal_quantile(alpha) / std * np.array([-1, 1])

        elif kind == "likelihood_ratio":
            # Compute the likelihood of the reference value (the MLE) and
            # a grid of values from which we construct the CI later
            # TODO Could be improved by, beginning from the MLE, search
            #      outwards where we are below the threshold, that method
            #      would probably be more precise
            a_grid = np.linspace(0, 1, num=10000)  # parameters to test
            logliks = np.array([self.loglik_wrapper(theta) for theta in a_grid])  # their log likelihood
            loglik_ref = self.loglik_wrapper(mle)  # reference value

            # Get indices of values that are above the loglik threshold
            chi_q = chi2_quantile(alpha)
            idcs = (logliks >= (loglik_ref - chi_q / 2))

            # Get the boundaries of the admitted values
            ci = np.append(np.min(a_grid[idcs]), np.max(a_grid[idcs]))
        else:
            raise AquaError("Confidence interval kind {} not implemented.".format(kind))

        self._ret[kind + "_ci"] = ci

        return self._ret
