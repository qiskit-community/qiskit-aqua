.. _oracles:

=======
Oracles
=======

Grover’s Search is a well known quantum algorithm for searching through
unstructured collections of records for particular targets with quadratic
speedups.

Given a set :math:`X` of :math:`N` elements
:math:`X=\{x_1,x_2,\ldots,x_N\}` and a boolean function :math:`f : X \rightarrow \{0,1\}`,
the goal on an *unstructured-search problem* is to find an
element :math:`x^* \in X` such that :math:`f(x^*)=1`.
Unstructured  search  is  often  alternatively  formulated  as  a  database  search  problem, in
which, given a database, the goal is to find in it an item that meets some specification.
The search is called *unstructured* because there are no guarantees as to how the
database is ordered.  On a sorted database, for instance, one could perform
binary  search  to  find  an  element in :math:`\mathcal{O}(\log N)` worst-case time.
Instead, in an unstructured-search problem, there is no  prior knowledge about the contents
of the database.  With classical circuits, there is no alternative but
to perform a linear number of queries to find the target element.
Conversely, Grover’s Search algorithm allows to solve the unstructured-search problem
on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries. 

All that is needed for carrying out a search is an oracle from Aqua's oracles library for
specifying the search criterion, which basically indicates a hit or miss
for any given record.  More formally, an *oracle* :math:`O_f` is an object implementing a boolean function
:math:`f` as specified above.  Given an input :math:`x \in X`, :math:`O_f` returns :math:`f(x)`.  The
details of how :math:`O_f` works are unimportant; Grover's search algorithm treats an oracle as a black
box.

.. topic:: Extending the Oracle Library

    Consistent with its unique  design, Aqua has a modular and
    extensible architecture. Algorithms and their supporting objects, such as oracles for Grover's Search Algorithm,
    are pluggable modules in Aqua.
    New oracles are typically installed in the ``qiskit_acqua/utils/oracles`` folder and derive from
    the ``Oracle`` class.  Aqua also allows for
    :ref:`aqua-dynamically-discovered-components`: new oracles can register themselves
    as Aqua extensions and be dynamically discovered at run time independent of their
    location in the file system.
    This is done in order to encourage researchers and
    developers interested in
    :ref:`aqua-extending` to extend the Aqua framework with their novel research contributions.

.. seealso::

    `Section :ref:`aqua-extending` provides more
    details on how to extend Aqua with new components.

Currently, Aqua provides the SATisfiability (SAT) oracle
implementation, which takes as input an SAT problem
specified as a formula in
`Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>__
and searches for solutions to that problem.
Essentially, a CNF is a conjunction of one or more clauses, where a clause is a disjunction of
one or more literals:

.. code:: python

    cnf : str

The Aqua SAT oracle implementation expects a CNF to be a ``str`` value assigned to
the ``cnf`` parameter.  The value must be encoded in
`DIMACS CNF
format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__.
Once it receives a CNF as an input, the SAT oracle constructs the corresponding quantum search circuit
for Grover's Search Algorithm to operate upon.

The following is an example of a CNF expressed in DIMACS CNF format:

.. code::

    c This is an example DIMACS 3-sat file with 3 satisfying solutions: 1 -2 3, -1 -2 -3, 1 2 -3.
    p cnf 3 5
    -1 -2 -3 0
    1 -2 3 0
    1 2 -3 0
    1 -2 -3 0
    -1 2 3 0

The first line, following the ``c`` character, is a comment.
The second line specifies that the CNF is over three boolean variables --- let us call them
:math:`x_1, x_2, x_3`, and contains five clauses.  The five clauses, listed afterwards,
are implicitly joined by the logical
``AND`` operator, :math:`\land`, while the variables in each clause, represented by their indices,
are implicitly disjoined by
the logical ``OR`` operator, :math:`lor`.  The :math:`-` symbol preceding a boolean variable index
corresponds to the logical ``NOT`` operator, :math:`lnot`.  Character ``0`` marks the end
of each clause.  Essentially, the code above corresponds to the following CNF:
:math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3) \land (x_1 \lor \lnot x_2 \lor x_3) \land
(x_1 \lor x_2 \lor \lnot x_3) \land (x_1 \lor \lnot x_2 \lor \lnot x_3) \land (\lnot x_1 \lor x_2 \lor x_3)`.

Examples showing how to use the Grover algorithm in conjunction with the SAT oracles to search
for solutions to SAT problems are available in the ``optimization`` folder of the
`Aqua Tutorials GitHub repository <https://github.com/Qiskit/aqua-tutorials>`__.

.. topic:: Declarative Name

   When referring to the SAT oracle declaratively inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it,
   is ``SAT``.

