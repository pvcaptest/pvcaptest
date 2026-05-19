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
   captest.validate_test_setup
   captest.resolve_test_setup
   captest.perc_wrap

.. _test-setups:

Predefined Test Setups
----------------------

:data:`~captest.captest.TEST_SETUPS` is a dict that maps preset names to
fully-validated test-setup entries. Each entry bundles a regression formula,
column mappings for measured and modeled data, default reporting conditions,
and a scatter-plot callable. Pass the preset name as ``test_setup`` when
constructing a :py:class:`~captest.captest.CapTest`.

.. data:: captest.TEST_SETUPS

   Registry of predefined capacity-test presets. Keys are preset-name strings;
   values are dicts with required keys ``description``, ``reg_cols_meas``,
   ``reg_cols_sim``, ``reg_fml``, ``scatter_plots``, and ``rep_conditions``.

The built-in presets are:

``e2848_default``
   Standard ASTM E2848 regression of AC power against front-side POA irradiance,
   ambient temperature, and wind speed using the full four-term formula. Default
   setup for monofacial capacity tests.

``bifi_e2848_etotal``
   Standard ASTM E2848 regression form with total effective irradiance replacing
   front-side POA as the independent variable.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi`, following the NREL
   modified bifacial approach.

``bifi_power_tc_meas_tbom``
   Temperature-corrected power regressed against front and rear irradiance.
   Back-of-module temperature is taken directly from field measurements and used
   to calculate cell temperature via the Sandia PV Array Performance Model.

``bifi_power_tc_calc_tbom``
   Temperature-corrected power regressed against front and rear irradiance.
   Back-of-module and cell temperature are both calculated from POA irradiance,
   ambient temperature, and wind speed via the Sandia model; no dedicated BOM
   sensor is required.

``e2848_spec_corrected_poa``
   Standard ASTM E2848 regression with a First Solar spectral correction applied
   to front-side POA before fitting. Requires relative humidity and station
   pressure on the measured side and precipitable water from the PVsyst output.
