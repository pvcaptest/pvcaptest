.. currentmodule:: captest

CapTest
=======

:py:class:`~captest.CapTest` organizes a pair of
:py:class:`~captest.capdata.CapData` objects (measured and simulated) along
with test configuration, and provides methods for computing reporting
conditions, running the ASTM E2848 capacity test, and evaluating pass/fail.

.. autosummary::
   :toctree: generated/

   CapTest

Constructors
------------

Alternative constructors for building a :py:class:`~captest.CapTest`
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
   CapTest.check_fit
   CapTest.resolved_setup
   CapTest.params
   CapTest.scatter_plots_name

Reporting Conditions
--------------------

A capacity test has a single set of reporting conditions, owned by the test as
:py:attr:`~captest.CapTest.rc` and tracked by
:py:attr:`~captest.CapTest.rc_source` (``'meas'`` / ``'sim'`` /
``'manual'``). Compute them with :py:meth:`~captest.CapTest.rep_cond`
or assign them directly via the ``rc`` setter. See :ref:`reporting_conditions`
in the user guide for the full model.

.. autosummary::
   :toctree: generated/

   CapTest.rc
   CapTest.rc_source
   CapTest.rep_cond
   CapTest.rep_irr_filter_low
   CapTest.rep_irr_filter_high

Test Settings
-------------

Parameters referenced from the prose elsewhere in the documentation. The full
list, with defaults, is in the :py:class:`~captest.CapTest` parameter table
above; see :ref:`captest-settings-reference` in the user guide for what each one
is used for.

.. autosummary::
   :toctree: generated/

   CapTest.rear_shade
   CapTest.auto_wrap_sim

Results
-------

Methods for running the capacity test and evaluating pass/fail.
:py:meth:`~captest.CapTest.run_test` runs the whole test — setup,
filter-pipeline replay, regressions, and results — in one call.
:py:meth:`~captest.CapTest.captest_results` returns a
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

Standalone functions used alongside :py:class:`~captest.CapTest`.

.. autosummary::
   :toctree: generated/

   load_config

.. currentmodule:: captest.captest

.. autosummary::
   :toctree: generated/

   test_setups
   resolve_test_setup
   load_presets
   perc_wrap

.. currentmodule:: captest

.. data:: captest.captest.SETUPS_DIR

   Directory of the preset documents shipped with the package
   (``captest/setups``); :py:func:`~captest.captest.load_presets` reads every
   ``*.yaml`` file in it to build :data:`~captest.captest.TEST_SETUPS`.

.. data:: captest.captest.SCATTER_REGISTRY

   Scatter-plot functions a setup's ``scatter_plots`` field may name:
   ``default`` (:py:func:`~captest.captest.scatter_default`), ``etotal``
   (:py:func:`~captest.captest.scatter_etotal`) and ``bifi_power_tc``
   (:py:func:`~captest.captest.scatter_bifi_power_tc`).

.. _test-setups:

Predefined Test Setups
----------------------

:data:`~captest.captest.TEST_SETUPS` is a dict that maps preset names to
validated test-setup documents. Each document bundles a regression formula,
regression-column trees for measured and modeled data, default reporting
conditions, the name of a scatter-plot function, and any test-level parameters
the setup requires. Pass the preset name as ``test_setup`` when constructing a
:py:class:`~captest.CapTest`.

.. data:: captest.captest.TEST_SETUPS

   Registry of predefined capacity-test presets. Keys are preset-name strings;
   values are :py:class:`captest.setup.TestSetup` documents loaded from
   ``setups/*.yaml`` (:data:`~captest.captest.SETUPS_DIR`) when ``captest`` is
   imported. Also available as ``captest.TEST_SETUPS``.

Every document carries a human-readable ``description`` summarizing the setup.
Read it programmatically with, e.g.,
``captest.TEST_SETUPS["bifi_e2848_etotal_rear_shade_sim"].description``. The
summaries below mirror those descriptions.

The built-in presets are:

``e2848_default``
   Standard ASTM E2848 regression of AC power against front-side POA irradiance,
   ambient temperature, and wind speed using the full four-term formula. Default
   setup for monofacial capacity tests.

``bifi_e2848_etotal_rear_shade_sim``
   Standard ASTM E2848 regression form with total effective irradiance replacing
   front-side POA as the independent variable. Rear shading and IAM losses are
   handled in the modeled (PVsyst) data (``rpoa_pvsyst = GlobBak + BackShd``)
   while the measured rear sensor is used as-measured, so ``rear_shade`` should
   be left at its default ``0``.
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
   as-measured, so ``rear_shade`` should be left at its default ``0``.
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
   measured rear sensor is used as-measured, so ``rear_shade`` should be left
   at its default ``0``.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi` with the spectral
   correction applied to :math:`E_{POA}`.

``bifi_e2848_etotal_rear_shade_meas_spec_corrected``
   Same regression form as ``bifi_e2848_etotal_rear_shade_sim_spec_corrected``,
   but rear-shading losses are applied on the measured side via the ``e_total``
   ``rear_shade`` factor while the modeled rear maps directly to PVsyst's
   unshaded ``GlobBak``.
   :math:`E_{Total} = E_{POA} + E_{Rear} \cdot \varphi \cdot (1 - s)` on the
   measured side, where :math:`s` is the ``rear_shade`` fraction.

.. warning::

   :py:attr:`~captest.CapTest.rear_shade` belongs with the
   ``*_rear_shade_meas`` presets. The measured ``reg_cols_meas`` mapping is the
   same in both variants, so a non-zero ``rear_shade`` would reach the measured
   ``e_total`` whichever preset is selected, and paired with a
   ``*_rear_shade_sim`` preset — where the shading is already carried by
   ``rpoa_pvsyst`` on the modeled side — it would double-count the loss. The
   ``*_rear_shade_sim`` presets therefore declare ``params: {rear_shade: 0}``,
   and :py:meth:`~captest.CapTest.setup` refuses a non-zero ``rear_shade``
   with them (see :py:attr:`~captest.CapTest.params`): it raises
   :py:class:`~captest.setup.SetupFitError`. Override ``params`` only if you
   deliberately want both.
