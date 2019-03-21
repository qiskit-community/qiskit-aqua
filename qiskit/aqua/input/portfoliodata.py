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


# Import requisite modules
import math
import operator
import logging
import datetime
import sys
import warnings

import numpy as np
import quandl
import fastdtw


class RandomData():

    def __init__(self, n):
        self.n = n
        # n are the inner variables

    def generate_instance(self):

        n = self.n
        np.random.seed(1543)

        xc = (np.random.rand(n) - 0.5) * 1
        yc = (np.random.rand(n) - 0.5) * 1

        rho = np.zeros([n, n])
        for ii in range(0, n):
            rho[ii,ii] = 1.
            for jj in range(ii + 1, n):
                rho[ii, jj] = 1/(1.+(xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2)
                rho[jj, ii] = rho[ii, jj]

        return xc, yc, -rho

# Get the data
class RealData():

    def __init__(self, n, plots = True):
        self.n = n
        self.plots = plots
        # n are the inner variables

    def generate_instance(self):

        n = self.n
        
        # Coordinates for visualisation purposes
        xc = np.zeros([n, 1])
        yc = np.zeros([n, 1])
        # The data series
        data = []        
        # The similarity measure
        rho = np.zeros([n, n])        

        # We will look at stock prices over the past year, starting at January 1, 2016
        start = datetime.datetime(2016,1,1)
        end = datetime.date.today()
        stocks = ["AAPL", "GOOG", "IBM"][:n]
        for (cnt, s) in enumerate(stocks):
            d = quandl.get("WIKI/" + s, start_date=start, end_date=end)
            xc[cnt, 1] = d["Adj. Close"][0]
            yc[cnt, 1] = d["Adj. Close"][-1]
            data.append(d["Adj. Close"])
            d.head()
            if self.plots: d["Adj. Close"].plot(grid = True, label=s)
        if self.plots:
            plt.legend()
            plt.title("Evolution of the adjusted closing price")
        
        for ii in range(0, n):
            rho[ii,ii] = 1.
            for jj in range(ii + 1, n):
                thisRho, path = fastdtw.fastdtw(data[ii], data[jj])
                rho[ii, jj] = thisRho
                rho[jj, ii] = rho[ii, jj]
        rho = rho / np.nanmax(rho)
        for ii in range(0, n):
            rho[ii,ii] = 1.

        if self.plots:
            fig, ax = plt.subplots()
            im = ax.imshow(rho)
            ax.set_title("Similarity measure based on pair-wise dynamic time warping")
            fig.tight_layout()
        return xc, yc, -rho
