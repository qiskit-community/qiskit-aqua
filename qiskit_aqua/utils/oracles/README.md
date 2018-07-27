# Qiskit Aqua - Oracles

Qiskit Algorithms for QUantum Applications (Qiskit Aqua) is a set of algorithms and utilities
for use with quantum computers. 
The *oracles* folder here contains oracle pluggable objects that may be used by algorithms

# Oracles

In quantum computing, an oracle refers to the part of an algorithm that can conceptually be treated as a black box.
It is oftentimes used in various quantum algorithms to achieve different goals. 
For example, an oracle used in the [Grover](../..#grover)'s Search algorithm specifies the search criterion.

The following oracles are supplied:

* [SAT](#sat)


## SAT

This oracle is for use when searching for solutions to a Satisfiability (SAT) problem. It implements the
[Conjunctive Normal Form](https://en.wikipedia.org/wiki/Conjunctive_normal_form) as used in specifying SAT problems,
and thus enables the provided Grover's implementation be used to build the search circuit for SAT problem instances.

* `cnf`=*string*

  A string in [DIMACS](http://www.satcompetition.org/2009/format-benchmarks2009.html) cnf format
  defining the SAT problem.


# Developers

New oracles may be added. See [Developers](../..#developers) section in algorithms readme
for further information.
