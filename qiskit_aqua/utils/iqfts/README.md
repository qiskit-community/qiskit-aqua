# QISKit AQUA - Inverse Quantum Fourier Tranforms (IQFT)

QISKit Algorithms and Circuits for QUantum Applications (QISKit AQUA) is a set of algorithms and utilities
for use with quantum computers. 
The *iqfts* folder here contains Inverse Quantum Fourier Transform pluggable objects that may be used by algorithms

# IQFTs

IQFTs are currently used by the by the [Quantum Phase Estimation (QPE)](../..#qpe) algorithm. 

The following IQFTs are supplied here:

* [STANDARD](#standard)
* [APPROXIMATE](#approximate)


## STANDARD

This is a standard IQFT. This IQFT has no parameters and will return an IQFT circuit based solely on
`num qubits`.


## APPROXIMATE

This is an approximate IQFT. This IQFT has the following parameters and will return an IQFT circuit based on
`num qubits`.

* `degree`=**integer, default is 0**

  This parameter controls the approximation. The value here will reduce the depth of neighbor terms allowed in the
  IQFT circuit. At 0 this will result in the same as the [STANDARD](#standard) IQFT. Each value above that however 
  reduces the range of the neighbor terms allowed by the corresponding amount thus reducing the circuit complexity.


# Developers

New IQFTs may be added. See [Developers](../..#developers) section in algorithms readme
for further information.
