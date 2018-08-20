.. _feature-extractions:

===================
Feature Extractions
===================

In machine learning, pattern recognition and image processing, *feature extraction*
starts from an initial set of measured data and builds derived values (also known as
*features*) intended to be informative and non-redundant, facilitating the subsequent
learning and generalization steps, and in some cases leading to better human
interpretations. Feature extraction is related to dimensionality reduction; it
involves reducing the amount of resources required to describe a large set of data.
When performing analysis of complex data, one of the major problems stems from the
number of variables involved. Analysis with a large number of variables generally
requires a large amount of memory and computation power, and may even cause a
classification algorithm to overfit to training samples and generalize poorly to new
samples.  When the input data to an algorithm is too large to be processed and is
suspected to be redundant (for example, the same measurement is provided in both
pounds and kilograms), then it can be transformed into a reduced set of features, named a *feature vector*. The process of determining a subset of the initial features is called *feature selection*. The selected features are expected to contain the relevant information from the input data, so that the desired task can
be performed by using the reduced representation instead of the complete initial data.

Aqua provides an extensible library of feature-extraction techniques, to be used in
:ref:`aqua-ai` and, more generally, in any quantum computing experiment that may
require constructing combinations of variables to get around the problems mentioned
above, while still describing the data with sufficient accuracy.

.. topic:: Extending the Feature Extraction Library

    Consistent with its unique  design, Aqua has a modular and
    extensible architecture. Algorithms and their supporting objects, such as
    feature extraction techniques for Artificial Intelligence,
    are pluggable modules in Aqua.
    New feature extraction are typically installed in the
    ``qiskit_aqua/utils/feature_extractions``
    folder and derive from the ``FeatureExtraction`` class.
    Aqua also allows for
    :ref:`aqua-dynamically-discovered-components`: new components can register themselves
    as Aqua extensions and be dynamically discovered at run time independent of their
    location in the file system.
    This is done in order to encourage researchers and
    developers interested in
    :ref:`aqua-extending` to extend the Aqua framework with their novel research contributions.