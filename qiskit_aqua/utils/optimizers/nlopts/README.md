# Qiskit Aqua - Optimizers - NLopt

Qiskit Algorithms for QUantum Applications (Qiskit Aqua) is a set of algorithms and utilities
for use with quantum computers. 
The *nlopts* folder here contains a variety of different optimizers for use by algorithms.
These optimizers are based on the open-source [NLOpt](https://nlopt.readthedocs.io) package.

# Optimizers

Optimizers may be used in variational algorithms, such as [VQE](../../..#vqe).
 
The following optimizers are supplied here. These are global optimizers:

* [CRS](#crs)
* [DIRECT_L](#direct_l)
* [DIRECT_L_RAND](#direct_l_rand)
* [ESCH](#esch)
* [ISRES](#isres)

These optimizers use the corresponding named optimizer from the open-source [NLopt](https://nlopt.readthedocs.io)
package. This package has native code implementations and must be installed locally for these optimizers here
to be used. See the [Installation](#installation) section below for more information. 

### Parameters

NLopt has the optimizers supported by a common interface. The parameters here are common to all the
optimizers provided below from the NLopt package and are:

* `max_evals`=*integer, default 1000*

  Maximum object function evaluations

## CRS

Controlled Random Search (CRS), with local mutation, for global optimization.

See NLopt http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#controlled-random-search-crs-with-local-mutation


## DIRECT_L

DIRECT is the DIviding RECTangles algorithm for global optimization, DIRECT_L is a *locally biased* variant

See NLopt http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#direct-and-direct-l


## DIRECT_L_RAND

This is as [DIRECT_L](#direct_l) above but with some randomization in decision evaluation of near-ties


## ESCH

ESCH is an evolutionary algorithm for global optimization

See NLopt http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#esch-evolutionary-algorithm


## ISRES

Improved Stochastic Ranking Evolution Strategy (ISRES) algorithm for non-linearly-constrained global optimization

See NLopt http://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#isres-improved-stochastic-ranking-evolution-strategy

# Installation

The [NLopt Download and installation](https://nlopt.readthedocs.io/en/latest/#download-and-installation)
instructions, and other more detailed installation information linked there, describe how to install NLopt.

If you running on Windows then you might like to also refer to
[NLopt on Windows](https://nlopt.readthedocs.io/en/latest/NLopt_on_Windows/)

## Hints

However in addition you may find the following hints helpful for Unix-like systems. First ensure your environment is set
to the python being used for which qiskit_aqua is installed and running. How having downloaded and unpacked the release
nlopt tar.gz file use the following commands

```
 > ./configure --enable-shared --with-python
 > make
 > sudo make install
```

The above makes and installs the shared libraries and python interface in `/usr/local`. To have these be used the
following can be done to augment the dynamic library load path and python path respectively, if you choose to leave 
these entities where they were built/installed as per above commands.

```
> export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
> export PYTHONPATH=/usr/local/lib/python3.6/site-packages:${PYTHONPATH}
```

Now you can run qiskit_aqua and these optimizers should be available for you to use.