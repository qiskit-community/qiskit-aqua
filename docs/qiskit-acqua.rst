QISKit ACQUA Overview
=====================

Problems that can benefit from the power of quantum computing
have been identified in numerous
domains, such as Chemistry, Artificial Intelligence (AI), Optimization
and Finance. Quantum computing, however, requires very specialized skills.
To address the needs of the vast population of practitioners who want to use and
contribute to quantum computing at various levels of the software stack, we have
created
QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA), an extensible library
of quantum algorithms that can be invoked directly or via domain-specific computational
applications: QISKit ACQUA Chemistry, QISKit ACQUA Artificial Intelligence, and QISKit ACQUA Optimization.

Modularity and Extensibility
----------------------------

QISKit ACQUA is the only end-to-end, cross-domain quantum software stack that allows for mapping high-level
classical computational software problems all the way down to a quantum machine (a simulator or a
real quantum device).

QISKit ACQUA was specifically designed in order to be extensible at each level of the software stack.
This allows different users with different levels of expertise and different scientific interests
to contribute to, and extend, the QISKit ACQUA software stack at different levels.

Input Generation
~~~~~~~~~~~~~~~~

At the application level, QISKit ACQUA allows for classical computational
software to be used as the quantum application front end.  This module is extensible;
new computational software can be easily plugged in.  Behind the scenes, QISKit ACQUA lets that
software perform some initial computations classically.  The  results of those computations are then combined with the problem
configuration and translated into input for one or more quantum algorithms, which invoke
the QISKit code APIs to build, compile and execute quantum circuits.

Input Translation
~~~~~~~~~~~~~~~~~

The problem configuration and (if present) the additional intermediate data
obtained from the classical execution of the computational software are
combined to form the input to the quantum system.  This phase, known as *translation*,
is also extensible.  Practitioners interested in providing more efficient
translation operators may do so by extending this layer of the QISKit ACQUA software
stack with their own translation operator implementation.

Quantum Algorithms
~~~~~~~~~~~~~~~~~~

Quantum algorithm researchers and developers can experiment with the algorithms already included
in QISKit ACQUA, or contribute their own algorithms via the pluggable interface exposed
by QISKit ACQUA.  In addition to plain quantum algorithms, QISKit ACQUA offers a vast eet
of supporting components, such as variational forms, local and global optimizers, initial states,
Quantum Fourier Transforms (QFTs) and Grover oracles.  These components are also extensible via pluggable
interfaces.

Novel Features
--------------

In addition to its modularity and extensibility, ability to span across mutiple
domains, and top-to-bottom completeness from classical computational software to
quantum hardware, compared to other quantum software stacks QISKit ACQUA present some unique advangates
in terms of usability, functionality, and configuration-correctness enforcement.  

User Experience
~~~~~~~~~~~~~~~

Allowing classical computational software at the front end has its own important advantages.
In fact, at the top of the QISKit ACQUA software stack are industry-domain experts, who are most likely very familiar with existing
computational software specific to their own domain.  These practitioners  may be interested
in experimenting with the benefits of quantum computing in terms of performance, accuracy
and reduction of computational complexity, but at the same time they might be
unwilling to learn about the underlying quantum infrastructure. Ideally,
such practitioners would like to use the computational software they are
used to as a front end to the quantum computing system, without having to learn a new quantum programming
language of new APIs.  It is also
likely that such practitioners may have collected, over time, numerous
problem configurations, corresponding to various experiments. QISKit ACQUA has been designed to accept those
configuration files  with no modifications, and
without requiring a practitioner experienced in a particular domain to
have to learn a quantum programming language. This approach has a clear advantage in terms
of usability.

Functionality
~~~~~~~~~~~~~

If QISKit ACQUA had been designed to interpose a quantum programming language
or new APIs between the user and the classical computational software, it would not have been able to
fully exploit all the features of the underlying classical computational software unless those features
had been exposed at the higher programming-language or API level.  In other words, in order to drive
the classical execution of any interfaced computational software to the most precise computation of the intermediate data needed to form
the quantum input, the advanced features of that software would have had to be configurable through QISKit ACQUA.
Given the intention for QISKit ACQUA to have an extensible interface capable of accepting any classical computational
software, the insertion of a quantum-specific programming language or API would have been not only a usability
obstacle, but also a functionality-limiting factor.
The ability of  QISKit ACQUA to directly interface classical computational software allows that software
to compute the intermediate data needed to form the quantum input at its highest level of precision.

Configuration Correctness
~~~~~~~~~~~~~~~~~~~~~~~~~

QISKit ACQUA offers another unique feature. Given that QISKit ACQUA
allows traditional software to be executed on a quantum system,
configuring an experiment in a particular domain may require a hybrid
configuration that involves both domain- and quantum-specific
configuration parameters. The chances of introducing configuration
errors, making typos, or selecting incompatible configuration parameters
are very high, especially for people who are expert in a given domain
but new to the realm of quantum computing. To address such issues, in
QISKit ACQUA the problem-specific configuration information and the
quantum-specific configuration information are dynamically verified for
correctness so that the combination of classical and quantum inputs is
resilient to configuration errors. Very importantly, configuration
correctness is dynamically enforced even for components that are
dynamically discovered and loaded,


Authors
-------

QISKit ACQUA was inspired, authored and brought about by the collective
work of a team of researchers.

QISKit ACQUA continues now to grow with the help and work of `many
people <CONTRIBUTORS.html>`__, who contribute to the project at different
levels.

License
-------

This project uses the `Apache License Version 2.0 software
license <https://www.apache.org/licenses/LICENSE-2.0>`__.

