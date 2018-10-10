.. _reciprocals:

===========
Reciprocals
===========

Aqua bundles methods to invert a fixed-point number prepared in a quantum register in the Reciprocals library.
Rather than being used as a standalone algorithm, the members of the library are to be used in a larger algorithm such as :ref:`HHL`. The following methods are available 

- :ref:`lookup`

- :ref:`gencircuit`

- :ref:`longdivision`

.. _lookup:

---------------------
Partial Table Look Up
---------------------

This method applies a variable sized binning to the values. Only a specified number of bits after the most-significant bit is taken into account when assigning rotation angles to the numbers prepared in the input register' states.
Using precomputed angles, the reciprocal is multiplied to the amplitude via controlled rotations.
While no resolution of the result is lost for small values, towards larger values the bin size increases. The accuracy of the result is tuned by the parameters. The following parameters are exposed:

- The number of bits used to approximate the numbers:

- The length of a sub string of the binary identifier:

- Switch for negative values:

  .. code::Python

     negative_evals = True | False

  If known beforehand that only positive values are present, one can set this switch to False and achieve a higher resolution in the output. The default is ``True``.

- The mimimum value present:

- The evolution time:

  .. code:: 

     evo_time : float

  This parameter scales the Eigenvalues in the :ref:`qpe_components` onto the range (0,1] ( (-0.5,0.5] for negativ EV ). If the Partial Table Look Up is used together with the QPE, the scale parameter can be estimated if the minimum EV and the evolution time are passed as parameters. The default is ``None``.


.. topic:: Declarative Name

   When referring to LookUp declaratively inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it, is ``LOOKUP``.

.. _gencircuit:

--------------------------
Generated Circuit Division
--------------------------

DESCRIPTION

The following parameters are exposed:

- The scale factor of the values:

-  The number of ancillae:

   .. code:: python

       num_ancillae = 5 | 6 | ...

 This parameter sets the number of ancillary qubits (the input register size).  A positive ``int`` value is expected.
   The default value is ``None`` and the minimum value ``5``.

- Switch for negative values:

  .. code::Python

     negative_evals = True | False

  If known beforehand that only positive values are present, one can set this switch to False and achieve a higher resolution in the output. The default is ``True``.

- The mimimum value present:

- The evolution time:

  .. code:: 

     evo_time : float

  This parameter scales the Eigenvalues in the :ref:`qpe_components` onto the range (0,1] ( (-0.5,0.5] for negativ EV ). If the Partial Table Look Up is used together with the QPE, the scale parameter can be estimated if the minimum EV and the evolution time are passed as parameters. The default is ``None``.

.. topic:: Declarative Name

   When referring to LookUp declaratively inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it, is ``GENCIRCUITS``.

.. _longdivision:

-------------
Long Division
-------------

.. topic:: Declarative Name

   When referring to LookUp declaratively inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it, is ``LongDivision``.
