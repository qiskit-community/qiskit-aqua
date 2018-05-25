# QISKit ACQUA

Quantum computing has the potential to solve problems that, due to their computational complexity, cannot be solved, either at 
all or for all practical purposes, on a classical computer.  On the other hand, Quantum computing requires very specialized 
skills.  Programming to a quantum machine is not an easy task, and requires specialized knowledge.  Problems that can benefit 
from the power of quantum computing, and for which no computationally affordable solution has been discovered in the general 
case on classical computers, have been identified in numerous domains, such as Chemistry, Artificial Intelligence (AI), 
Optimization and Finance.

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a library of algorithms allows practitioners with
different types of expertise to contribute to the Quantum software stack at different levels.

Industry-domain experts, who are most likely very familiar with existing computational software specific to their own domain, 
may be interested in the benefits of Quantum computing in terms of performance, accuracy and computational complexity, but at 
the same time they might be unwilling to learn about the underlying Quantum infrastructure.  Ideally, such practitioners would 
like to use the computational software they are used to as a front end to the Quantum computing system.  It is also likely 
that such practitioners may have collected, over time, numerous problem configurations corresponding to various experiments.  
In such cases, it would be desirable for a system that enables classical computational software to run on a Quantum 
infrastructure, to accept the same configuration files used classically, with no modifications, and without requiring a 
practitioner experienced in a particular domain to have to learn a Quantum programming language.

QISKit Algorithms and QISKit Applications allow computational software specific to any domain to be executed on a Quantum 
computing machine.  The computational software is used both as a form of domain-specific input specification and a form of 
Quantum-specific input generation.  The specification of the computational problem may be defined using the classical 
computational software.  The classical computational software may be executed classically to extract some additional 
intermediate data necessary to form the input to the Quantum system.  And finally, the problem configuration and (if present) 
the additional intermediate data obtained from the classical execution of the computational software are combined to form the 
input to the Quantum system.

In order to form the input to the Quantum machine, the input coming from the classical computational software and the 
user-provided configuration needs to be translated.  The translation layer is domain- and problem-specific.  For example, in 
order to compute some molecular properties, such as the ground-state molecular energy, dipole moment and excited states of a 
molecule, QISKit ACQUA for Chemistry translates the classically computed input into a Fermionic Hamiltonian and from that it
will generate a Qubit Hamiltonian, which will then be passed to a Quantum algorithm in the QISKit ACQUA library for the energy
computation.  Viable algorithms that can solve these problems quantumly include Variational Quantum Eigensolver (VQE) and
Quantum Phase Estimation (QPE).

The Quantum algorithm in the QISKit ACQUA forms the circuits to be executed by a Quantum device or simulator.  The 
major novelty of QISKit ACQUA is that the applications running on top of it allow for classical computational software to be
used without having to be wrapped around a common infrastructure,  The users of QISKit ACQUA will not have to learn a new
programming paradigm, and they will be still able to use the computational software they are used to.

A novel characteristic of QISKit ACQUA is that it allows researchers, developers and practitioners with different types 
of expertise to contribute at different levels of the QISKit ACQUA stack, such as the Hamiltonian-
generation layer, and the algorithm layer (which includes, among other things, Quantum algorithms, optimizers, variational 
forms, and initial states).

A unique feature of QISKit ACQUA is that the software stack is applicable to different domains, 
such as Chemistry, Artificial Intelligence and Optimization.  QISKIT ACQUA is a common infrastructure among 
the various domains, and the application layers built on top of QISKit ACQUA library are all structured according to the 
same architecture.  New domains can be added easily, taking advantage of the shared Quantum algorithm infrastructure, and new 
algorithms and algorithm components can be plugged in and automatically discovered at run time via dynamic lookups.

QISKit ACQUA offers another unique feature.  Given that QISKit ACQUA allows 
traditional software to be executed on a Quantum system, configuring an experiment in a particular domain may require a hybrid 
configuration that involves both domain-specific and Quantum-specific configuration parameters.  The chances of introducing 
configuration errors, making typos, or selecting incompatible configuration parameters are very high, especially for people 
who are expert in a given domain but new to the realm of Quantum computing.  To address such issues, in QISKit ACQUA
the problem-specific configuration information and the Quantum-specific configuration information are dynamically verified for 
correctness so that the combination of classical and Quantum inputs is resilient to configuration errors.  Very 
importantly, configuration correctness is dynamically enforced even for components that are dynamically discovered and loaded, 
which includes traditional computational software packages, input translation modules, algorithms, variational forms, 
optimizers, and initial states.

In essence, QISKit ACQUA is a novel software framework that allows users to experience the flexibility provided by the
integration of classical computational software, the error-resilient configuration, the ability to contribute new
components at different levels of the Quantum software stack, and the ability to extend QISKit ACQUA to new domains.
