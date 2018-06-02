# QISKit ACQUA - Initial States

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a set of algorithms and utilities
for use with quantum computers. 
The *initial_states* folder here contains initial-state pluggable objects that may be used by algorithms

# Initial States

Initial states are currently used to define a starting state for the [variational forms](../variational_forms/README.md)
and as a trial state to be evolved by the [Quantum Phase Estimation (QPE)](../../../qiskit_acqua#qpe) algorithm.
An initial state provides a circuit that can take the starting point of all zero qubits to the defined state.  

The following initial states are supplied:

* [ZERO](#zero)
* [CUSTOM](#custom)
* [HartreeFock](#hartreefock)


## ZERO

This initial state is for when the all-zeroes state is the desired starting state. This is the case for a vacuum state in
physics/chemistry. This initial state has no parameters and will return the zero state based solely on
`num_qubits`.


## CUSTOM

This initial state allows for different desired starting states. In addition to some pre-defined states, a totally 
custom state may be specified through this option. This initial state has the following parameters and will return
a state for use with `num_qubits`.

* `state`=**zero** | uniform | random

  This state can take the value of *zero*, which is the same as using the [ZERO](#zero) initial state object,
  *uniform* which will set the qubits in superposition, or *random*.

* `state_vector`=*array of numbers*

  The *state_vector* parameter allows a specific custom initial state to be defined. The state vector must be an
  array of numbers of length `2 ** num_qubits`. The vector will be normalized so that the total probability represented
  is 1.0.


## HartreeFock

This initial state corresponds to the Hartree-Fock state for a molecule in chemistry. The following parameters allow
this initial state to be configured:

* `qubit_mapping`=jordan_wigner | **parity** | bravyi_kitaev

  The mapping that is used from fermion to qubit. Note: bravyi_kitaev is also known as binary-tree-based qubit mapping. 

* `two_qubit_reduction`=**true** | false

  Whether two-qubit reduction is being used. Only valid with parity mapping, otherwise ignored
  
* `num_particles`=*integer*

  The number of particles for which the state should be created 

* `num_orbitals`=*integer*

  The number of spin orbitals for which the state should be created 


# Developers

New initial states may be added. See [Developers](../../../qiskit_acqua#developers) section in algorithms readme
for further information.
