.. currentmodule:: captest

CapTest
=======

:py:class:`~captest.captest.CapTest` organizes a pair of
:py:class:`~captest.capdata.CapData` objects (measured and simulated) along
with test configuration, and provides methods for computing reporting
conditions, running the ASTM E2848 capacity test, and evaluating pass/fail.

.. autosummary::
   :toctree: generated/

   CapTest

Constructors
------------

Alternative constructors for building a :py:class:`~captest.captest.CapTest`
from parameters, YAML files, or mapping objects.

.. autosummary::
   :toctree: generated/

   CapTest.from_params
   CapTest.from_yaml
   CapTest.from_mapping

Setup
-----

Methods for configuring the test, re-loading data, and serializing
configuration. ``setup`` and ``reload`` accept a ``side`` argument
(``'meas'`` / ``'sim'`` / ``'both'``) to act on one side only.

.. autosummary::
   :toctree: generated/

   CapTest.setup
   CapTest.reload
   CapTest.to_yaml
   CapTest.to_mapping
   CapTest.resolved_setup

Reporting Conditions
--------------------

A capacity test has a single set of reporting conditions, owned by the test as
:py:attr:`~captest.captest.CapTest.rc` and tracked by
:py:attr:`~captest.captest.CapTest.rc_source` (``'meas'`` / ``'sim'`` /
``'manual'``). Compute them with :py:meth:`~captest.captest.CapTest.rep_cond`
or assign them directly via the ``rc`` setter. See :ref:`reporting_conditions`
in the user guide for the full model.

.. autosummary::
   :toctree: generated/

   CapTest.rc
   CapTest.rep_cond
   CapTest.rep_irr_filter_low
   CapTest.rep_irr_filter_high

Results
-------

Methods for running the capacity test and evaluating pass/fail.
:py:meth:`~captest.captest.CapTest.run_test` runs the whole test — setup,
filter-pipeline replay, regressions, and results — in one call.
:py:meth:`~captest.captest.CapTest.captest_results` returns a
:py:class:`~captest.captest.CapTestResults` object; ``str(results)``
reproduces the printed report and ``results.styled_pvalues()`` the styled
p-value table.

.. autosummary::
   :toctree: generated/

   CapTest.run_test
   CapTest.captest_results
   CapTest.captest_results_check_pvalues
   CapTest.determine_pass_or_fail
   CapTest.get_summary
   captest.captest.CapTestResults

Visualization
-------------

.. autosummary::
   :toctree: generated/

   CapTest.scatter_plots
   CapTest.overlay_scatters
   CapTest.residual_plot

Module-level Functions
----------------------

Standalone functions used alongside :py:class:`~captest.captest.CapTest`.

.. autosummary::
   :toctree: generated/

   load_config
   captest.captest.test_setups
   captest.captest.validate_test_setup
   captest.captest.resolve_test_setup
   captest.captest.perc_wrap

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

Every entry carries a human-readable ``description`` key summarizing the setup.
Read it programmatically with, e.g.,
``captest.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim"]["description"]``. The
summaries below mirror those ``description`` strings.

The built-in presets are:

``e2848_default``
   Standard ASTM E2848 regression of AC power against front-side POA irradiance,
   ambient temperature, and wind speed using the full four-term formula. Default
   setup for monofacial capacity tests.

``bifi_e2848_etotal_rear_shade_sim``
   Standard ASTM E2848 regression form with total effective irradiance replacing
   front-side POA as the independent variable. Rear shading and IAM losses are
   handled in the modeled (PVsyst) data (``rpoa_pvsyst = GlobBak + BackShd``)
   while the measured rear sensor is used as-measured (``rear_shade = 0``).
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi`, following the NREL
   modified bifacial approach. See :ref:`choosing-test-setup` in the CapTest
   user guide for the per-side formulas.

``bifi_e2848_etotal_rear_shade_meas``
   Same regression form as ``bifi_e2848_etotal_rear_shade_sim``, but rear-shading
   losses are applied on the measured side via the ``e_total`` ``rear_shade``
   factor while the modeled rear maps directly to PVsyst's unshaded ``GlobBak``.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi \cdot (1 - s)` on the
   measured side, where :math:`s` is the ``rear_shade`` fraction.

``bifi_power_tc_meas_tbom``
   Temperature-corrected power regressed against front and rear irradiance.
   Back-of-module temperature is taken directly from field measurements and used
   to calculate cell temperature via the Sandia PV Array Performance Model.

``bifi_power_tc_calc_tbom``
   Temperature-corrected power regressed against front and rear irradiance.
   Back-of-module and cell temperature are both calculated from POA irradiance,
   ambient temperature, and wind speed via the Sandia model; no dedicated BOM
   sensor is required.

``bifi_power_tc_etotal_rear_shade_sim``
   Temperature-corrected power regressed against total effective irradiance.
   Back-of-module temperature is taken from field measurements and used to
   calculate cell temperature via the Sandia PV Array Performance Model. Rear
   shading and IAM losses are handled in the modeled (PVsyst) data
   (``rpoa_pvsyst = GlobBak + BackShd``) while the measured rear sensor is used
   as-measured (``rear_shade = 0``).
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi`, following the NREL
   modified bifacial approach.

``bifi_power_tc_etotal_rear_shade_meas``
   Same regression form as ``bifi_power_tc_etotal_rear_shade_sim``, but
   rear-shading losses are applied on the measured side via the ``e_total``
   ``rear_shade`` factor while the modeled rear maps directly to PVsyst's
   unshaded ``GlobBak``.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi \cdot (1 - s)` on the
   measured side, where :math:`s` is the ``rear_shade`` fraction.

``e2848_spec_corrected_poa``
   Standard ASTM E2848 regression with a First Solar spectral correction applied
   to front-side POA before fitting. Requires relative humidity and atmospheric
   pressure on the measured side and precipitable water from the PVsyst output.

``bifi_e2848_etotal_rear_shade_sim_spec_corrected``
   Standard ASTM E2848 regression with total effective irradiance replacing
   front-side POA and a First Solar spectral correction applied to the
   front-side POA used to calculate the total irradiance. Requires relative
   humidity and atmospheric pressure on the measured side and precipitable
   water from the PVsyst output. Rear shading and IAM losses are handled in
   the modeled (PVsyst) data (``rpoa_pvsyst = GlobBak + BackShd``) while the
   measured rear sensor is used as-measured (``rear_shade = 0``).
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi` with the spectral
   correction applied to :math:`E_{POA}`.

``bifi_e2848_etotal_rear_shade_meas_spec_corrected``
   Same regression form as ``bifi_e2848_etotal_rear_shade_sim_spec_corrected``,
   but rear-shading losses are applied on the measured side via the ``e_total``
   ``rear_shade`` factor while the modeled rear maps directly to PVsyst's
   unshaded ``GlobBak``.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi \cdot (1 - s)` on the
   measured side, where :math:`s` is the ``rear_shade`` fraction.
