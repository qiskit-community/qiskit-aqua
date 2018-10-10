.. _eigs:

====
Eigs
====

Aqua bundles methods to find Eigenvalue of a given matrix, such as :ref:`qpe_components` in the Eigs library.
Rather than being used as a standalone algorithm, the members of the library are to be used in a larger algorithm such as :ref:`HHL`. The following methods are available 

- :ref:`qpe_component`: Given a matrix and a linear combination of its eigenstates, *qpe* prepares the Eigenvalues on a specified output register. 


.. topic:: Extending the Eigs Library

    Consistent with its unique  design, Aqua has a modular and
    extensible architecture. Algorithms and their supporting objects, such as optimizers for quantum variational algorithms,
    are pluggable modules in Aqua.
    New eigenvalue solver  are typically installed in the ``qiskit_aqua/algorithms/components/eigs`` folder and derive from
    the ``Eigenvalues`` class.  Aqua also allows for
    :ref:`aqua-dynamically-discovered-components`: new optimizers can register themselves
    as Aqua extensions and be dynamically discovered at run time independent of their
    location in the file system.
    This is done in order to encourage researchers and
    developers interested in
    :ref:`aqua-extending` to extend the Aqua framework with their novel research contributions.


.. seealso::

    `Section :ref:`aqua-extending` provides more
    details on how to extend Aqua with new components.

.. _qpe_component:

---
QPE
---
This Eigenvalue solver component is directly based on the QPE quantum algorithm in aqua :ref:`qpe`.
Some changes have been made to support negative Eigenvalues and use it in a larger quantum algorithm (e.g. :ref:`hhl`).

.. seealso::

    `Section :ref:`qpe` provides more
    details on the QPE algorithm.


In addition to requiring an IQFT and an initial state as part of its
configuration, QPE also exposes the following parameter settings:

-  The number of time slices:

   .. code:: python

       num_time_slices = 0 | 1 | ...

   This has to be a non-negative ``int`` value.  The default value is ``1``.

-  Paulis grouping mode:

   .. code:: python

       paulis_grouping = "default" | "random"

   Two string values are permitted: ``"default"`` or ``"random"``, with ``"random"``
   being the default and indicating that ???.

-  The expansion mode:

   .. code:: python

       expansion_mode = "trotter" | "suzuki"

   Two ``str`` values are permitted: ``"trotter"`` (Lloyd's method) or ``"suzuki"`` (for Trotter-Suzuki expansion),
   with  ``"trotter"`` being the default one.

-  The expansion order:

   .. code:: python

       expansion_order = 1 | 2 | ...

   This parameter sets the Trotter-Suzuki expansion order.  A positive ``int`` value is expected.  The default value is ``1``.

-  The number of ancillae:

   .. code:: python

       num_ancillae = 1 | 2 | ...

   This parameter sets the number of ancillary qubits to be used by QPE.  A positive ``int`` value is expected.
   The default value is ``1``.

- The evolution time:

  .. code:: 

     evo_time : float

  This parameter scales the EV onto the range (0,1] ( (-0.5,0.5] for negativ EV ). If not provided, it is calculated internally by using an estimation of the highest EV present in the matrix. The default is ``None``.

- Switch for negative Eigenvalues:

  .. code::Python

     negative_evals = True | False

  If known beforehand that only positive EV are present, one can set this switch to False and achieve a higher resolution in the output. The default is ``True``.

- Switch for non-hermitian input:

  .. code::python

     hermitian_matrix = True | False

  If non-hermitian is selected, a hermitian matrix of size 2 input size is used as an input and the result gives the singular values of the matrix. The default is ``True``.

- Switch for the usage of basis gates:

  .. code::python

     use_basis_gates = True | False

  Passed to the construction routine of the evolution circuit used in QPE. The default is ``True``.

.. topic:: Declarative Name

   When referring to QPE declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``QPE``.

