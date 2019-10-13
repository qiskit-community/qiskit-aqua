# Final Project Description

VQE is a computing algorithm that can be used to obtain the eigenvalues of a Hamiltonian. A significant aspect of it involves finding the expectation values of Pauli strings. A N-qubit Hamitonian will require us to solve a worst case of 4^N Pauli strings which is an exponential problem. To speed this process up, we can form groups of Pauli strings that commute with each other together such that we only need to solve for any one of the strings to obtain information on all of them in the group. Following which, we can further optimise Qiskit's VQE which supports measurements on the z-basis by working out an algorithm to effectively rotate all the Pauli strings into the z-basis to conduct measurements all in one go as well.

Currently, Qiskit-Aqua has a module that can determine the commutability of Pauli strings and then form a Pauli graph. However, we have found a tighter condition that correctly checks for some cases that were left out in the initial code.

To test the code out, navigate to qiskit/aqua/operators/ and replace __init__.py, pauli_graph.py 
as well as add in pauli_graph2.py attached in this folder. Run the Python notebook file. 
