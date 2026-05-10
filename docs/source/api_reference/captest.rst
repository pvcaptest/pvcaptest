.. currentmodule:: captest

CapTest
=======

:py:class:`~captest.captest.CapTest` organizes a pair of
:py:class:`~captest.capdata.CapData` objects (measured and simulated) along
with test configuration, and provides methods for computing reporting
conditions, running the ASTM E2848 capacity test, and evaluating pass/fail.

.. autosummary::
   :toctree: generated/

   captest.CapTest

Constructors
------------

Alternative constructors for building a :py:class:`~captest.captest.CapTest`
from parameters, YAML files, or mapping objects.

.. autosummary::
   :toctree: generated/

   captest.CapTest.from_params
   captest.CapTest.from_yaml
   captest.CapTest.from_mapping

Setup
-----

Methods for configuring the test and serializing configuration.

.. autosummary::
   :toctree: generated/

   captest.CapTest.setup
   captest.CapTest.to_yaml
   captest.CapTest.resolved_setup

Reporting Conditions
--------------------

.. autosummary::
   :toctree: generated/

   captest.CapTest.rep_cond
   captest.CapTest.rep_irr_filter_low
   captest.CapTest.rep_irr_filter_high

Results
-------

Methods for running the capacity test and evaluating pass/fail.

.. autosummary::
   :toctree: generated/

   captest.CapTest.captest_results
   captest.CapTest.captest_results_check_pvalues
   captest.CapTest.determine_pass_or_fail
   captest.CapTest.get_summary

Visualization
-------------

.. autosummary::
   :toctree: generated/

   captest.CapTest.scatter_plots
   captest.CapTest.overlay_scatters
   captest.CapTest.residual_plot

Module-level Functions
----------------------

Standalone functions used alongside :py:class:`~captest.captest.CapTest`.

.. autosummary::
   :toctree: generated/

   captest.load_config
