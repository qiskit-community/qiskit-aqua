#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:27:10 2018

@author: gawel
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_res_and_theory(res, A, k, nege):
    def theory(y, w, k, n, t):
        r = np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y)))/
            (1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))
        r[np.isnan(r)] = 2**k
        r = 2**(-2*k-n)*r**2
        r/=sum(r)
        return r

    x = []
    y = []
    for c, _, l in res["measurements"]:
        x += [l]
        y += [c]

    ty = np.arange(0, 2**k, 1)
    w, v = np.linalg.eigh(A)
    n = len(A)
    
    data = theory(ty, w.real, k, n, res["evo_time"])

    if nege:
        tx = np.arange(0, 2**k, 1)/2**k
        tx[2**(k-1):] = -(1-tx[2**(k-1):])
        tx *= 2*np.pi/res["evo_time"]
        tx =   np.concatenate((tx[2**(k-1):], tx[:2**(k-1)])) 
        data = np.concatenate((data[2**(k-1):], data[:2**(k-1)])) 
    else:
        tx = np.arange(0, 2**k, 1)/2**k
        tx *= 2*np.pi/res["evo_time"]


    plt.bar(x, y, width=2*np.pi/res["evo_time"]/2**k)
    plt.plot(tx, data, "r")

    plt.show()
   
def initiate_q_backend():   
    import sys, time, getpass
    try:
        sys.path.append("../../") # go to parent dir
        import Qconfig
        qx_config = {
            "APItoken": Qconfig.APItoken,
            "url": Qconfig.config['url']}
        print('Qconfig loaded from %s.' % Qconfig.__file__)
    except:
        APItoken = getpass.getpass('Please input your token and hit enter: ')
        qx_config = {
            "APItoken": APItoken,
            "url":"https://quantumexperience.ng.bluemix.net/api"}
        print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')    
        
    #from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
  
    from qiskit import available_backends, execute, register, get_backend, registered_providers, unregister ,compile
    from qiskit.backends.local.localprovider import LocalProvider
    
    for provider in registered_providers():
        if not isinstance(provider, LocalProvider):
            unregister(provider)
    #from qiskit.tools.visualization import plot_histogram
    #from pprint import pprint   
    
    register(qx_config['APItoken'], qx_config['url'])
    
    print(available_backends())
    

def Efr(res):
    Eigs = np.array(res['measurements'])[:2,2]
    Eigs = Eigs.astype(float)
    return Eigs


 
