# Chemistry In Qiskit

Qiskit's chemistry module is a set of tools, algorithms and software for use with quantum computers
to carry out research and investigate how to take advantage of quantum computing power to solve chemistry
problems.

If you need introductory material see the main [readme](../../README.md) which has
[installation](../../README.md#installation) instructions and information on how to use the
chemisty module for
[running a chemistry experiment](../../README.md#running-a-chemistry-experiment).

This readme contains the following sections:

* [Developers](#developers)
* [Additional reading](#additional-reading)

## Developers

### Result dictionary

The ChemistryOperator process_algorithm_result() method returns a result dictionary.
Energies are in units of `Hartree` and dipole moment in units of `a.u.`.

The dictionary contains the following fields of note:

* *energy*

  The ground state energy

* *energies*

  An array of energies comprising the ground state energy and any excited states if they were computed

* *nuclear_repulsion_energy*

  The nuclear repulsion energy

* *hf_energy*

  The Hartree-Fock ground state energy as computed by the driver

* *nuclear_dipole_moment*, *electronic_dipole_moment*, *dipole_moment*

  Nuclear, electronic and combined dipole moments for x, y and z

* *total_dipole_moment*

  Total dipole moment

* *algorithm_retvals*

  The result dictionary of the algorithm that ran for the above values. See the algorithm for any further information.

### For writers of algorithms and other utilities such as optimizers and variational forms:

Qiskit Aqua is the library of cross-domain algorithms and utilities. Please refer to the documentation
there for more information on how to write and contribute such objects to Qiskit Aqua. Such objects are then available
to be used by Qiskit's chemistry module.

### For unit test writers:

Unit tests should go under "test" folder and be classes derived from QiskitChemistryTestCase class.

They should not have print statements, instead use self.log.debug. If they use assert, they should be from the unittest
package like self.AssertTrue, self.assertRaises etc.

For guidance look at the tests cases implemented at https://github.com/Qiskit/qiskit-sdk-py/tree/master/test/python


### For unit test running:

To run all unit tests: `python -m unittest discover`

To run a particular unit test module: `python -m unittest test/chemistry/test_end2end_with_iqpe.py`

For help: `python -m unittest -h`

There are other running options at: https://docs.python.org/3/library/unittest.html#command-line-options

In order to see unit test log messages you need to set environment variable:
```
LOG_LEVEL=DEBUG
export LOG_LEVEL
```

The example above will save all results from "self.log.debug()" to a ".log" file with same name as the module used to
run. For instance "test_end2end.log" in the test folder.

## Additional Reading

Here are some references to other useful materials that may be helpful

* [Quantum optimization using variational algorithms on near-term quantum devices](https://arxiv.org/abs/1710.01022)
* [Fermionic Quantum Computation](https://arxiv.org/abs/quant-ph/0003137v2)

