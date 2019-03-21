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
import matplotlib.pyplot as plt


# Get the data
class StockMarketData():
    def __init__(self, tickers):
        self.n = len(tickers)
        self.tickers = tickers
        # the raw data, stored as one row per ticker
        self.data = []
        # covariance matrix
        self.cov = np.zeros([self.n, self.n])    
        # The similarity measure
        self.rho = np.zeros([self.n, self.n])    
        # n are the inner variables

    def load_random(self, start = datetime.datetime(2016,1,1), end = datetime.datetime(2016,1,30)):
        np.random.seed(1543)
        self.data = []  
        days = (end - start).days
        self.data = np.random.rand(len(self.tickers), days)

    # Loads data from Wikipedia via the quadnl package. Please see quandl T&C. 
    def load_wikipedia(self, start = datetime.datetime(2016,1,1), end = datetime.datetime(2016,1,30)):
        n = self.n
        # The data series
        self.data = []        
        try:
          import quandl
          for (cnt, s) in enumerate(self.tickers):
            d = quandl.get("WIKI/" + s, start_date=start, end_date=end)
            self.data.append(d["Adj. Close"])
            d.head() 
        except ImportError:
            print("This requires the quandl module and an internet connection.")   

    # Loads data from the official Nasdaq Data on Demand service, cf. https://www.nasdaqdod.com/ 
    # This requires an access token provided by Nasdaq
    def load_Nasdaq(self, token, start = datetime.datetime(2016,1,1), end = datetime.datetime(2016,1,30)):
        import re
        import urllib
        import urllib2
        import xml.etree.cElementTree as ElementTree
        url = 'http://ws.nasdaqdod.com/v1/NASDAQAnalytics.asmx/GetSummarizedTrades'
        values = {'_Token' : token,
          'Symbols' : ','.join(self.tickers),
          'StartDateTime' : start.strftime("%m/%d/%Y %H:%M:%S.%f") , 
          'EndDateTime' : end.strftime("%m/%d/%Y %H:%M:%S.%f") , 
          'MarketCenters' : '' ,
          'TradePrecision': 'Day', # Hour, ...
          'TradePeriod':'1'
        }
        request_parameters = urllib.urlencode(values)
        req = urllib2.Request(url, request_parameters)
        try: 
            response = urllib2.urlopen(req)
            root = ElementTree.XML(response.read())
            # extraction ... 
        except: print("Accessing Nasdaq failed.")

    # gets coordinates suitable for plotting
    def get_coordinates(self):
        # Coordinates for visualisation purposes
        xc = np.zeros([self.n, 1])
        yc = np.zeros([self.n, 1])
        xc = (np.random.rand(self.n) - 0.5) * 1
        yc = (np.random.rand(self.n) - 0.5) * 1
        #for (cnt, s) in enumerate(self.tickers):
        #xc[cnt, 1] = self.data[cnt][0]
        # yc[cnt, 0] = self.data[cnt][-1]
        return xc, yc

    def get_covariance(self):
        if not self.data: return None
        self.cov = np.cov(self.data, rowvar = True)
        return self.cov

    def get_similarity_matrix(self):
        if not self.data: return None    
        try:
          import fastdtw
          for ii in range(0, self.n):
            self.rho[ii,ii] = 1.
            for jj in range(ii + 1, self.n):
                thisRho, path = fastdtw.fastdtw(self.data[ii], self.data[jj])
                self.rho[ii, jj] = thisRho
                self.rho[jj, ii] = self.rho[ii, jj]
          self.rho = self.rho / np.nanmax(self.rho)
          for ii in range(0, self.n):
            self.rho[ii,ii] = 1.
        except ImportError:
          print("This requires fastdtw package.")
        return self.rho

    def plot(self):  
        #for (cnt, s) in enumerate(self.tickers):
        #    plot(self.data[cnt], grid = True, label=s)
        #plt.legend()
        #plt.title("Evolution of the adjusted closing price")
        #plt.show()
        self.get_covariance()
        self.get_similarity_matrix()
        print("Top: a similarity measure. Bottom: covariance matrix.")
        plt.subplot(211)
        plt.imshow(self.rho)
        plt.subplot(212)
        plt.imshow(self.cov)
        plt.show()     

