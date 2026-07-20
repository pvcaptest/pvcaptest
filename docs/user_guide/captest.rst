.. _captest:

CapTest Workflow
================
:py:class:`~captest.captest.CapTest` is a convenient way to keep the measured
data, modeled data, test settings, and comparison plots together for one
capacity test.

The general workflow is still the same workflow described in :ref:`dataload`: load the
data, review the column groups, filter the measured and modeled data, calculate
reporting conditions, fit regressions, and compare the results. ``CapTest``
helps with the pieces that are repeated from project to project:

- Keeping the measured and modeled datasets together as ``ct.meas`` and
  ``ct.sim``.
- Applying a named regression setup, such as the standard ASTM E2848 equation
  or one of the bifacial options.
- Storing common test values, such as nameplate capacity, per-inverter AC
  nameplate (``inv_ac_nameplate``), test tolerance, irradiance filter limits,
  shade-filter settings, and bifaciality.
- Creating comparison plots and pass/fail summaries from the same test object.
- Reading and writing the test setup from a yaml file, which can be helpful
  when you want a repeatable project record.

A ``CapTest`` object keeps the test level requirements (e.g., minimum irradiance
and test tolerance) for the test in a single place. The raw measured data and PVsyst
data remain in the associated ``CapData`` objects, while the test-level assumptions
are stored on ``CapTest``.

When to use CapTest
-------------------
Almost always! The goal of the CapTest class is to streamline conducting tests whose
regression equation includes regressors that are values calculated from the data
available, like :math:`E_{Total}` for a bifacial test. 

If there is not a test setup available that fits the regression equation you are using,
then you can create your own test setup and, if necessary, functions to create the
calculated parameters. See the :ref:`custom_test_setups` section for details on this
process. Also, please open an issue on Github to request adding a new test setup!

The examples in :ref:`dataload` use :py:class:`~captest.capdata.CapData`
directly. That workflow is mostly unchanged and may be helpful while learning pvcaptest or while working through a non-standard analysis.

``CapTest`` becomes more helpful when:

- You want to use one of the standard regression setups without manually
  assigning a regression columns dictionary and then calling ``process_regression_columns`` to recursively process it.
- You want one place to store project-level assumptions such as AC nameplate,
  tolerance, bifaciality, and filter settings.
- You plan to save the test setup in a yaml file and re-run it later.
- You want comparison methods such as
  :py:meth:`~captest.captest.CapTest.captest_results`,
  :py:meth:`~captest.captest.CapTest.overlay_scatters`, and
  :py:meth:`~captest.captest.CapTest.residual_plot` to use the same measured
  and modeled data automatically.

.. _choosing-test-setup:

Choosing a test setup
---------------------
The ``test_setup`` value tells pvcaptest which regression equation and default
measured/model column mappings to use. These setups are intended to cover common
capacity-test cases without requiring users to create one.

A test setup is a named preset that bundles everything needed to configure
the regression for a capacity test. Each setup defines:

- **Regression formula** — the model equation, such as the standard ASTM E2848
  four-term formula or the bifacial temperature-corrected power formula (see
  :ref:`identifying-regression-data`).
- **Measured column mappings** (``reg_cols_meas``) — which measured data columns
  map to each regression variable, how multiple sensors are aggregated (sum or
  mean), and any calculated columns required by the setup (e.g. ``e_total``,
  ``power_temp_correct``).
- **Modeled column mappings** (``reg_cols_sim``) — the corresponding PVsyst output
  columns and any calculated columns for the modeled side.
- **Default reporting conditions** — how each regression variable is aggregated to
  compute reporting conditions (e.g. 60th-percentile POA, mean ambient temperature
  and wind speed).
- **Scatter plot function** — the plotting callable matched to the regression
  formula, used by :py:meth:`~captest.captest.CapTest.scatter_plots`.

See :ref:`custom_test_setups` for additional details and example.

The complete list of built-in presets, with a description of each, is in
:ref:`test-setups` in the API reference. You can also print the available
setup names at any time by calling :py:func:`~captest.captest.test_setups`
(pass ``descriptions=True`` to also print each setup's summary):

.. code-block:: Python

    >>> import captest as ct
    >>> ct.test_setups()
    All options
    ============================================================
    e2848_default
    bifi_e2848_etotal_rear_shade_sim
    bifi_e2848_etotal_rear_shade_meas
    bifi_power_tc_meas_tbom
    bifi_power_tc_calc_tbom
    ...
    bifi_e2848_etotal_rear_shade_meas_spec_corrected

The most commonly used options are described below:

``e2848_default``
    Standard ASTM E2848 regression:

    .. math::
        P = E_{POA}\left(a_{1} + a_{2} E_{POA} + a_{3} T_{a} + a_{4} v\right)

    This is the default setup for monofacial capacity tests.

``bifi_e2848_etotal_rear_shade_sim``
    Uses the standard ASTM E2848 regression form, but replaces front-side POA
    with total irradiance. Rear shading and IAM losses are handled in the
    modeled (PVsyst) data: the modeled rear irradiance is
    ``rpoa_pvsyst = GlobBak + BackShd``, while the measured rear sensor
    (``irr_rpoa``) is used as-measured (``rear_shade = 0``).

    .. math::
        E_{Total}^{model} = E_{POA} + \left(GlobBak + BackShd\right)\varphi

    .. math::
        E_{Total}^{meas} = E_{POA} + E_{Rear}\,\varphi

    where :math:`\varphi` is the bifaciality factor.

``bifi_e2848_etotal_rear_shade_meas``
    Same regression form as ``bifi_e2848_etotal_rear_shade_sim``, but rear
    shading is applied on the measured side as a flat fraction :math:`s` (the
    ``rear_shade`` factor), while the modeled rear maps directly to PVsyst's
    unshaded global rear (``GlobBak``).

    .. math::
        E_{Total}^{model} = E_{POA} + GlobBak \cdot \varphi

    .. math::
        E_{Total}^{meas} = E_{POA} + E_{Rear}\,\varphi\left(1 - s\right)


    where :math:`\varphi` is the bifaciality factor and :math:`s` is the
    ``rear_shade`` fraction. 

``bifi_power_tc_meas_tbom``
    Uses temperature-corrected power as the dependent variable and regresses it
    against front and rear irradiance. Back-of-module temperature is taken
    directly from field measurements and used to calculate cell temperature
    via the Sandia PV Array Performance Model.

``bifi_power_tc_calc_tbom``
    Same regression as ``bifi_power_tc_meas_tbom`` but back-of-module
    temperature is calculated from POA irradiance, ambient temperature, and
    wind speed rather than measured directly. Both setups create a two-panel
    scatter plot so the front- and rear-side relationships can be reviewed
    separately.

``e2848_spec_corrected_poa``
    Uses the standard ASTM E2848 regression form, but applies a First Solar
    spectral correction to POA irradiance before fitting the regression. This
    setup requires humidity and pressure data on the measured side and
    precipitable water from the PVsyst output. See :ref:`spec_corrected_poa`
    for the additional inputs.

The remaining built-in presets combine the pieces above — total-irradiance
variants of the temperature-corrected power setups
(``bifi_power_tc_etotal_rear_shade_sim``,
``bifi_power_tc_etotal_rear_shade_meas``) and spectrally corrected
total-irradiance setups
(``bifi_e2848_etotal_rear_shade_sim_spec_corrected``,
``bifi_e2848_etotal_rear_shade_meas_spec_corrected``). See :ref:`test-setups`
in the API reference for their descriptions.

.. note::

    The built-in setup names are strings. For example,
    ``test_setup='e2848_default'`` uses the standard ASTM E2848 setup, and
    ``test_setup='bifi_e2848_etotal_rear_shade_sim'`` uses the total-irradiance
    bifacial setup.

.. note:: 

    Each built-in test setup maps its regression variables to specific
    **column group IDs** — the string keys of the ``column_groups`` attribute.
    The IDs hardcoded into the built-in setups are:

    - ``irr_poa`` — front-side plane-of-array irradiance (all setups)
    - ``irr_rpoa`` — rear-side plane-of-array irradiance (all bifacial setups)
    - ``temp_bom`` — back-of-module temperature (temperature-corrected setups
      using measured BOM temperature: ``bifi_power_tc_meas_tbom``,
      ``bifi_power_tc_etotal_rear_shade_sim``,
      ``bifi_power_tc_etotal_rear_shade_meas``)
    - ``real_pwr_mtr`` — AC power meter (all setups)
    - ``temp_amb`` — ambient temperature (all setups)
    - ``wind_speed`` — wind speed (all setups)
    - ``humidity`` — relative humidity (spectrally corrected setups)
    - ``pressure`` — station pressure (spectrally corrected setups)

    .. warning::

        If your data uses different column group IDs (for example because your
        column-group YAML template assigns a different name to your irradiance
        sensor), the built-in setup will not find the expected groups and the
        regression will fail or use incorrect data. In that case you must either
        rename the column groups in your data to match the IDs above, or supply
        a fully custom ``reg_cols_meas`` that references your actual column
        group IDs. See :ref:`custom_test_setups` for details.

Creating a CapTest
------------------
A :py:class:`~captest.captest.CapTest` can be created from from file paths, data
that has already been loaded, or from a yaml file. Using ``from_params`` will create
a CapTest object given file paths and is the option recommended for typical usage of
pvcaptest to interactivley run a test in a Jupyter notebook.

From data paths
~~~~~~~~~~~~~~~
If you provide paths to your data, ``CapTest`` will load the data for you. 

.. code-block:: Python

    ct = CapTest.from_params(
        test_setup='bifi_e2848_etotal_rear_shade_sim',
        meas_path='./data/measured/',
        sim_path='./data/pvsyst_results.csv',
        bifaciality=0.15,
        ac_nameplate=6_000_000,
        test_tolerance='- 4',
        meas_load_kwargs={
            'group_columns': './path-to/column_groups.xlsx',
        },
    )

Measured data is loaded with :py:func:`~captest.io.load_data`, and modeled data
is loaded with :py:func:`~captest.io.load_pvsyst`. Extra loading options can be
passed with ``meas_load_kwargs`` and ``sim_load_kwargs``.

.. note::

   You will likely need / want to include meas_load_kwargs to load your
   column grouping from a file. See the examples.

From loaded data
~~~~~~~~~~~~~~~~
If you have already loaded the measured and modeled data, pass the two
``CapData`` objects to :py:meth:`~captest.captest.CapTest.from_params`.

.. code-block:: Python

    from captest import CapTest, load_data, load_pvsyst

    meas = load_data(path='./data/measured/')
    sim = load_pvsyst(path='./data/pvsyst_results.csv')

    ct = CapTest.from_params(
        test_setup='e2848_default',
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance='- 4',
    )
    ct.setup()

.. note::

    Note, the last line that calls ``setup()``. When manually constructing a 
    CapTest object as shown here this is a necessary step. See :ref:`what-setup-does`.

The measured data is then available as ``ct.meas`` and the modeled data is
available as ``ct.sim``. Both are regular ``CapData`` objects, so the filtering,
plotting, reporting-condition, and regression methods used elsewhere in the
User Guide still apply.

From yaml
~~~~~~~~~
For repeatable project work, the capacity-test setup can be stored in a yaml
file and loaded with :py:meth:`~captest.captest.CapTest.from_yaml`.

.. code-block:: yaml

    captest:
      test_setup: bifi_e2848_etotal_rear_shade_sim
      meas_path: ./data/measured/
      sim_path: ./data/pvsyst.csv
      ac_nameplate: 6_000_000
      test_tolerance: "- 4"
      min_irr: 400
      max_irr: 1400
      fshdbm: 1.0
      bifaciality: 0.15

.. code-block:: Python

    ct = CapTest.from_yaml('./project.yaml')

``from_yaml`` loads the measured and modeled data and runs
:py:meth:`~captest.captest.CapTest.setup`. Filter pipelines saved in the file
(``meas_filters`` / ``sim_filters``) are **not** applied at load — they are
stored on the instance as ``ct.meas_filters_pending`` and
``ct.sim_filters_pending`` and run later by
:py:meth:`~captest.captest.CapTest.run_test`. The
:ref:`captest-typical-workflow` section walks through the resulting states
step by step. Pass ``run_setup=False`` to skip ``setup()`` as well and load
the data only (see :ref:`load-only`).

Relative ``meas_path`` and ``sim_path`` values are interpreted relative to the
yaml file location. This makes the yaml file portable with the project folder.

One yaml file can also contain more than one capacity-test setup. For example,
the same bifacial project may be reviewed with both the total-irradiance E2848
setup and the temperature-corrected power setup.

.. code-block:: yaml

    captest_bifi_etotal:
      test_setup: bifi_e2848_etotal_rear_shade_sim
      meas_path: ./data/measured/
      sim_path: ./data/pvsyst.csv
      ac_nameplate: 6_000_000
      bifaciality: 0.15

    captest_bifi_power_tc:
      test_setup: bifi_power_tc_calc_tbom
      meas_path: ./data/measured/
      sim_path: ./data/pvsyst.csv
      ac_nameplate: 6_000_000
      bifaciality: 0.15

.. code-block:: Python

    ct_etotal = CapTest.from_yaml('./project.yaml', key='captest_bifi_etotal')
    ct_power_tc = CapTest.from_yaml('./project.yaml', key='captest_bifi_power_tc')

.. _what-setup-does:

What setup does
---------------
When ``CapTest`` has both measured and modeled data, it prepares each
``CapData`` object for the selected test setup. This happens automatically when
using :py:meth:`~captest.captest.CapTest.from_params` or
:py:meth:`~captest.captest.CapTest.from_yaml` with both datasets present,
unless ``run_setup=False`` is passed (see :ref:`load-only`).

The setup step:

- Assigns the regression equation for the selected ``test_setup``.
- Assigns the measured and modeled columns used by that equation.
- Aggregates sensors where the setup calls for a sum or average.
- Creates calculated columns required by the setup, such as ``e_total`` for
  the total-irradiance bifacial setup or ``power_temp_correct`` for the
  temperature-corrected bifacial setup.
- Copies scalar values such as ``bifaciality``, ``power_temp_coeff``,
  ``base_temp``, and ``spectral_module_type`` onto the measured and modeled
  ``CapData`` objects so calculated columns use the intended assumptions.

If you change a setup value after creating ``ct``, call
:py:meth:`~captest.captest.CapTest.setup` again before continuing.

.. note::

    Calling ``setup()`` resets ``ct.meas.data_filtered`` and
    ``ct.sim.data_filtered`` back to the unfiltered data. This is usually what
    you want after changing the setup, but it also means filters should be
    re-applied after calling ``setup()`` again.

.. _captest-typical-workflow:

The typical workflow, step by step
----------------------------------
This section walks the typical notebook workflow — from a saved config file to
results — and spells out the state of the ``CapTest`` after each step. The
same methods apply when a test is being built for the first time; the only
difference is that the filter steps are created interactively (see
:ref:`building-the-pipeline`) instead of being replayed from the config file.

Step 1 — load the test
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    from captest import CapTest

    ct = CapTest.from_yaml('./project.yaml')

The measured and modeled data are loaded and
:py:meth:`~captest.captest.CapTest.setup` runs: scalar settings are propagated
to ``ct.meas`` / ``ct.sim``, calculated columns (e.g. ``e_total``) are
created, and the regression columns are resolved and aggregated. The filter
pipelines saved in the file are stored as pending — they have not touched the
data.

.. note::

    **State after this step.**

    - ``ct.meas.data`` / ``ct.sim.data`` hold the loaded (plus calculated)
      columns, and ``data_filtered`` equals ``data`` — nothing is filtered.
    - The applied filter chains (``ct.meas.filters`` / ``ct.sim.filters``)
      are empty; the config's pipelines are held in
      ``ct.meas_filters_pending`` / ``ct.sim_filters_pending``.
    - ``ct.rc`` is ``None`` when the config's ``rc_source`` is a computed
      source (``'meas'`` or ``'sim'``). For ``rc_source: manual`` the
      ``reporting_conditions_values`` are validated and seeded during
      ``setup()``, so ``ct.rc`` is set and ``ct.rc_source == 'manual'``.
    - No regressions are fitted (``regression_results`` is ``None`` on both
      sides).

Step 2 — review the raw data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because nothing is filtered yet, the review tools show the full dataset:

.. code-block:: Python

    ct.meas.plot()
    ct.scatter_plots()

.. note::

    **State after this step.** Unchanged — reviewing does not modify the
    test.

Step 3 — run the measured side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    ct.run_test(side='meas')

This runs the measured side's setup, replays the pending measured pipeline,
and fits the measured regression. The modeled side is untouched.

.. note::

    **State after this step.**

    - ``ct.meas.filters`` holds the applied chain and
      ``ct.meas.data_filtered`` is the filtered data;
      ``ct.meas_filters_pending`` is empty (consumed by the run).
    - ``ct.rc`` is set by the pipeline's ``RepCond`` step
      (``rc_source == 'meas'`` for a measured-source config).
    - ``ct.meas.regression_results`` holds the fitted measured regression.
    - ``ct.sim`` still holds unfiltered data with its pipeline pending;
      nothing is fitted on the modeled side.

Step 4 — summarize the measured filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    ct.meas.get_summary()
    print(ct.meas.describe_filters())
    ct.meas.scatter_filters() + ct.meas.timeseries_filters()

Step 5 — adjust the filtering (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To append a filter, call any ``filter_*`` method — it runs immediately at the
tail of the applied chain:

.. code-block:: Python

    ct.meas.filter_time(start='2026-03-26', end='2026-04-12')

To change a filter in the middle of the chain, edit the applied step's
parameters and re-run from its position with
:py:meth:`~captest.capdata.CapData.rerun_filters_from`; to insert, delete, or
reorder steps, use the serialize–edit–replay workflow described in
:ref:`editing-filter-pipeline`. Re-run the summaries afterwards. If an edit
changes the reporting conditions, a warning names any applied
``ref_val='rep_irr'`` steps that resolved against the previous conditions so
they can be re-run (see :ref:`reporting_conditions`).

.. note::

    **State after this step.** The measured chain and
    ``ct.meas.data_filtered`` reflect the edits. The measured regression fit
    is stale until it is re-fitted (step 7, or another
    ``run_test(side='meas')``).

Step 6 — run and summarize the modeled side
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    ct.run_test(side='sim')
    ct.sim.get_summary()

Modeled filters that anchor on the reporting irradiance
(``ref_val='rep_irr'``) resolve against the test reporting conditions
``ct.rc`` established in step 3.

.. note::

    **State after this step.** Both chains are applied, both pending lists
    are consumed, both regressions are fitted, and ``ct.rc`` is set.

Step 7 — results
~~~~~~~~~~~~~~~~
When both fits are current, compare them directly:

.. code-block:: Python

    results = ct.captest_results()

If filters were adjusted after the last per-side run, re-fit first
(``ct.meas.fit_regression()`` / ``ct.sim.fit_regression()``) — or simply
re-run the whole test from the applied chains in one call:

.. code-block:: Python

    results = ct.run_test()

``str(results)`` (or ``print(results)``) is the test report. See
:ref:`reviewing-results` for the fields of the returned
:py:class:`~captest.captest.CapTestResults`.

.. _building-the-pipeline:

Building the filter pipeline interactively
------------------------------------------
When a test is being put together for the first time there is no config file
to replay — the filter pipeline is built by calling the filtering methods on
``ct.meas`` and ``ct.sim`` directly, in the same way separate ``CapData``
objects are used.

The example below shows the general pattern. Actual filters should be selected
to match the contract and test procedure.

.. code-block:: Python

    # measured filters
    ct.meas.filter_irr(ct.min_irr, ct.max_irr)
    ct.meas.filter_outliers()
    ct.rep_cond()
    ct.meas.fit_regression()

    # simulated data filters
    ct.sim.filter_time(start='2026-03-26', end='2026-04-12')
    ct.sim.filter_irr(ct.min_irr, ct.max_irr)
    ct.sim.filter_pvsyst()
    ct.sim.fit_regression()

    results = ct.captest_results()

:py:meth:`~captest.captest.CapTest.rep_cond` calculates reporting conditions
using the selected setup's defaults. For the standard E2848 setup, POA is
calculated using the 60th percentile of filtered POA, while ambient temperature
and wind speed use the mean.

If reporting conditions should be calculated from modeled data instead of
measured data, use:

.. code-block:: Python

    ct.rep_cond(which='sim')

After reporting conditions are calculated, it is common to apply a second,
narrower irradiance filter around the reporting irradiance.
``ct.rep_irr_filter_low`` and ``ct.rep_irr_filter_high`` provide the lower and
upper fractional bounds. With the default ``rep_irr_filter=0.2``, these values
are ``0.8`` and ``1.2``. Using these attributes of the ``CapTest`` instance helps
to apply these consistently in the filtering of the measured and simulated data.

.. code-block:: Python

    ct.meas.filter_irr(
        ct.rep_irr_filter_low,
        ct.rep_irr_filter_high,
        ref_val='rep_irr',
    )

    ct.sim.filter_irr(
        ct.rep_irr_filter_low,
        ct.rep_irr_filter_high,
        ref_val='rep_irr',
    )

The ``ref_val='rep_irr'`` argument resolves the reference irradiance from the
single test reporting conditions (``ct.rc``), so the measured and modeled
filters anchor on the same value. See :ref:`reporting_conditions` for the full
reporting-conditions model.

Once the pipelines are in place, save the whole test — settings and both
pipelines — with :py:meth:`~captest.captest.CapTest.to_yaml` so it can be
reproduced later (see :ref:`saving_reproducing`).

.. _running-with-run-test:

Running the whole test with run_test
------------------------------------
:py:meth:`~captest.captest.CapTest.run_test` runs the complete test in one
call. It runs :py:meth:`~captest.captest.CapTest.setup`, replays each side's
filter pipeline (the ``rc_source`` side first, so its reporting-conditions
step establishes ``ct.rc`` before the other side's RC-dependent filters
resolve), fits both regressions, verifies the reporting conditions were
computed during the run, and returns the results.

.. code-block:: Python

    results = ct.run_test()
    results.cap_ratio

Combined with :py:meth:`~captest.captest.CapTest.from_yaml`, this reproduces a
whole capacity test — data loading, setup, filtering, reporting conditions,
regressions, and results — from a single config file, with each pipeline
executed exactly once (the load stores the pipelines pending; ``run_test``
replays them):

.. code-block:: Python

    results = CapTest.from_yaml('./project.yaml').run_test()

For each side, ``run_test`` chooses the pipeline it replays:

- the **applied chain**, when ``cd.filters`` is non-empty — interactive edits
  always win. The chain is snapshotted with
  :py:meth:`~captest.capdata.CapData.filters_to_config` before ``setup()``
  clears it, which is what makes ``run_test`` re-entrant: calling it again
  after adjusting a test-level parameter re-runs the same pipeline with the
  new settings;
- otherwise, the side's **pending pipeline** (``ct.meas_filters_pending`` /
  ``ct.sim_filters_pending``, stored by ``from_yaml`` / ``from_mapping``).

A side's pending list is consumed once its replay succeeds — after a test has
run, the applied chain is the single source of truth, and a later
``reset_filter()`` + ``run_test()`` means "no filters", not "restore the
config's filters". If a step fails during a replay, the pipeline is rolled
back to its pre-call state and the failed side's pipeline definition is
retained in ``ct.<side>_filters_pending``; see :ref:`replay-failure` for the
recovery loop.

Re-running one side
~~~~~~~~~~~~~~~~~~~
Pass ``side='meas'`` or ``side='sim'`` to re-run only one side's setup,
filter pipeline, and regression, leaving the other side untouched. Per-side
runs return the ``CapTest`` instance itself rather than results, so a full
comparison still ends with :py:meth:`~captest.captest.CapTest.captest_results`.

This pairs well with :py:meth:`~captest.captest.CapTest.reload`, which
re-loads one side's data from its stored path with the stored loader and
keyword arguments and re-runs the per-side setup. For example, after dropping
an updated PVsyst export into the project folder, refresh and re-run just the
modeled side:

.. code-block:: Python

    ct.reload('sim').run_test(side='sim')
    results = ct.captest_results()

.. note::

    ``reload`` requires the ``CapTest`` to have been constructed from data
    paths (``from_params`` with ``meas_path`` / ``sim_path``, ``from_yaml``,
    or ``from_mapping``); it raises a ``ValueError`` when the instance was
    built from pre-loaded ``CapData`` objects.

.. _load-only:

Loading without running setup
-----------------------------
Pass ``run_setup=False`` to :py:meth:`~captest.captest.CapTest.from_params`,
:py:meth:`~captest.captest.CapTest.from_yaml`, or
:py:meth:`~captest.captest.CapTest.from_mapping` to load the data and stop
there — for example to inspect the raw data or review the column groups
before the regression mapping is applied:

.. code-block:: Python

    ct = CapTest.from_yaml('./project.yaml', run_setup=False)

.. note::

    **State after this call.** ``ct.meas`` and ``ct.sim`` hold the loaded
    data, but nothing ``setup()`` produces exists yet: no scalar
    propagation, no calculated columns, no regression-column processing or
    aggregation. Pending pipelines (and, for a manual-RC config, the pending
    reporting-conditions values) are stored for later. A subsequent
    ``ct.setup()`` or ``ct.run_test()`` proceeds normally from this state.

.. _editing-filter-pipeline:

Editing the filter pipeline
---------------------------
Appending a filter is direct — every ``filter_*`` method runs immediately at
the tail of the applied chain. Editing the *middle* of a pipeline — changing
a step's settings, inserting, deleting, or reordering steps — uses the
pipeline's serialized form: get the pipeline as a list of plain dicts, edit
the list, and replay it.

.. code-block:: Python

    cfg = ct.meas.filters_to_config()      # list of dicts, one per step

    cfg[1]['low'] = 300                    # change a setting
    del cfg[2]                             # drop a step
    cfg.insert(1, {'type': 'Time', 'start': '2026-03-26', 'end': '2026-04-12'})

    ct.meas.run_pipeline(cfg)              # re-run the edited pipeline

:py:meth:`~captest.capdata.CapData.run_pipeline` always resets the applied
chain and rebuilds it from the config — replay is restore-then-re-run — so
the result is exactly the edited pipeline, in order. To review a pipeline
before replaying it, dump it as yaml:

.. code-block:: Python

    import yaml
    print(yaml.safe_dump(cfg, sort_keys=False))

When the edit only changes parameters of steps already in the chain,
:py:meth:`~captest.capdata.CapData.rerun_filters_from` avoids the
serialization round-trip: edit the live step object and re-run from its
position.

.. code-block:: Python

    ct.meas.filters[2].low = 300
    ct.meas.rerun_filters_from(2)

Which list do I edit?
~~~~~~~~~~~~~~~~~~~~~
- **Applied chain** (``cd.filters`` non-empty): start from
  ``cd.filters_to_config()`` and replay with ``cd.run_pipeline(cfg)``.
- **Pending pipeline** (just loaded from a config file, or after a failed
  ``run_test`` replay): edit ``ct.meas_filters_pending`` /
  ``ct.sim_filters_pending`` in place — that list *is* the pipeline
  ``run_test`` will run. ``filters_to_config()`` cannot serve here because
  the applied chain holds none of the pending steps.

Choosing a re-run method
~~~~~~~~~~~~~~~~~~~~~~~~
- :py:meth:`~captest.capdata.CapData.rerun_filters_from` — replay-only, one
  side, from a chain position onward; picks up live edits to the applied
  steps' parameters; no setup, no regression fit.
- ``ct.run_test(side='meas')`` / ``ct.run_test(side='sim')`` — one side's
  setup, full pipeline replay, and regression fit.
- ``ct.run_test()`` — the whole test; returns
  :py:class:`~captest.captest.CapTestResults`.

.. _replay-failure:

When a replay fails
~~~~~~~~~~~~~~~~~~~
``run_pipeline`` and ``rerun_filters_from`` are transactional: if any step
raises, the filter chain, the steps' runtime state, and the reporting
conditions (``rc`` / ``rc_tool`` on the ``CapData``, and the test-level
``ct.rc`` / ``ct.rc_source``) are restored to their pre-call values before
the exception propagates, and a note naming the failing step is attached to
the error (Python 3.11+). A partially applied pipeline is never left behind.

When the failure happens inside ``run_test``, the failed side's pipeline
definition is retained in ``ct.<side>_filters_pending`` and the error's note
points there. The recovery loop is:

1. Read the error — it names the failing step (position and type).
2. Edit that step's dict in ``ct.meas_filters_pending`` (or
   ``ct.sim_filters_pending``).
3. Re-run with ``ct.run_test()`` — the rollback left the chain empty, so the
   corrected pending pipeline is selected again — or replay directly with
   ``ct.meas.run_pipeline(ct.meas_filters_pending)``.
4. Repeat until the pipeline runs clean. Alternatively, fix the yaml config
   file and reload it with ``from_yaml``.

Recipe: switching the reporting-conditions source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A test that computed its reporting conditions from the measured data can be
re-run with modeled reporting conditions (or vice versa) using the editing
workflow — no dedicated API is needed.

The cleanest route is through the config file: edit the yaml so that
``rc_source: sim``, move the ``RepCond`` entry from ``meas_filters`` to
``sim_filters`` (placing it before any ``ref_val: rep_irr`` step), then
reload and run:

.. code-block:: Python

    results = CapTest.from_yaml('./project.yaml').run_test()

In-session, the same move is the edit workflow applied to both pipeline
configs:

.. code-block:: Python

    meas_cfg = ct.meas.filters_to_config()
    sim_cfg = ct.sim.filters_to_config()

    # move the RepCond dict from the measured list to the modeled list
    rep_cond_step = next(s for s in meas_cfg if s['type'] == 'RepCond')
    meas_cfg.remove(rep_cond_step)
    sim_cfg.insert(2, rep_cond_step)   # before any ref_val: rep_irr step

    ct.sim.run_pipeline(sim_cfg)    # RepCond flips ct.rc / rc_source to 'sim'
    ct.meas.run_pipeline(meas_cfg)  # rep_irr filters resolve against new ct.rc

    ct.meas.fit_regression()
    ct.sim.fit_regression()
    results = ct.captest_results()

Run the modeled side first: its ``RepCond`` step computes the new reporting
conditions and flips ``ct.rc`` / ``ct.rc_source`` to ``'sim'`` — the
source-change ``UserWarning`` at that point is expected and confirms the
switch. Two other warnings guide wrong-order mistakes:

- If both pipelines contain a ``RepCond`` step (e.g. the measured copy was
  not removed), the dual-``RepCond`` warning flags the configuration as
  ambiguous — whichever side runs second would overwrite the test reporting
  conditions and flip ``rc_source`` again.
- If the measured pipeline is replayed first, its ``ref_val='rep_irr'``
  filter resolves against the *old* reporting conditions; when the modeled
  ``RepCond`` then changes ``ct.rc``, the staleness warning names the
  measured steps that no longer match so they can be re-run — for example
  with ``ct.meas.rerun_filters_from(...)``.

See :ref:`reporting_conditions` for the full description of both warnings.

.. _reviewing-results:

Reviewing results
-----------------
The main comparison method is
:py:meth:`~captest.captest.CapTest.captest_results`. It predicts the measured
and modeled capacities at the reporting conditions, calculates the capacity
ratio, and prints a pass/fail summary using the AC nameplate and test
tolerance stored on your instance of Captest, e.g. ``ct``. It returns a
:py:class:`~captest.captest.CapTestResults` object holding the individual
result values:

.. code-block:: Python

    results = ct.captest_results()
    results.cap_ratio            # capacity ratio, actual / expected
    results.passed               # pass/fail against the test tolerance
    results.expected_capacity    # modeled output at reporting conditions
    results.actual_capacity      # measured output at reporting conditions

``str(results)`` (or ``print(results)``) reproduces the printed report, and
``results.styled_pvalues()`` returns a styled table of the regression
coefficients and p-values for both sides with high p-values highlighted.
Other fields include the p-value-checked capacity ratio
(``cap_ratio_pval_check``), the tolerance and capacity bounds the test was
judged against, the points remaining after filtering on each side
(``points_used``), per-side regression tables, and the reporting conditions
used with their provenance (``rc``, ``rc_source``).

Headline selection with check_pvalues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``captest_results`` always computes two variants of the predicted capacities:
the plain predictions from the fitted coefficients, and the
**p-value-checked** predictions, in which any coefficient whose p-value is
above the ``pval`` cutoff (default ``0.05``) is set to zero before
predicting. The ``check_pvalues`` argument selects which variant is the
*headline* — the values reported as ``cap_ratio``, ``actual_capacity``,
``expected_capacity``, and ``tested_capacity`` and used for the pass/fail
decision:

.. code-block:: Python

    results = ct.captest_results()                    # headline = plain
    results = ct.captest_results(check_pvalues=True)  # headline = checked

Two fields make the selection unambiguous regardless of how the test was
run:

- ``results.cap_ratio_pval_check`` always carries the p-value-checked ratio,
  even when the headline is the plain variant.
- ``results.pvalues_checked`` records which variant the headline fields hold
  (``True`` when the test ran with ``check_pvalues=True``).

The same arguments are accepted by
:py:meth:`~captest.captest.CapTest.run_test`, which forwards them to
``captest_results``.

.. note::

    Versions before v0.17.0 returned the capacity ratio as a bare float from
    ``captest_results``. Code that used the return value directly should now
    read ``results.cap_ratio``.

Additional review methods are available from the same ``CapTest`` object:

- :py:meth:`~captest.captest.CapTest.captest_results_check_pvalues` compares
  results with and without high-p-value coefficients and highlights
  coefficients with p-values above 0.05.
- :py:meth:`~captest.captest.CapTest.overlay_scatters` overlays the measured
  and modeled regression scatter plots.
- :py:meth:`~captest.captest.CapTest.residual_plot` compares measured and
  modeled residuals against the regression variables.
- :py:meth:`~captest.captest.CapTest.get_summary` combines the filter summaries
  for ``ct.meas`` and ``ct.sim`` into one table.
- :py:meth:`~captest.captest.CapTest.determine_pass_or_fail` applies the stored
  nameplate and tolerance to a capacity ratio.

Scatter plots
-------------
:py:meth:`~captest.captest.CapTest.scatter_plots` creates the scatter plot that
matches the selected ``test_setup``. By default it plots the measured data.

.. code-block:: Python

    ct.scatter_plots()
    ct.scatter_plots(which='sim')

The built-in scatter plots support several options that are useful during data
review.

AM/PM split
~~~~~~~~~~~
Use ``split_day=True`` to show morning and afternoon points separately.

.. code-block:: Python

    ct.scatter_plots(split_day=True)

By default pvcaptest tries to determine the split time from modeled clear-sky
GHI, when that information is available. Otherwise it uses ``"12:30"``. To set
the split time manually, pass ``split_time``.

.. code-block:: Python

    ct.scatter_plots(split_day=True, split_time='12:45')

The marker and color styles can be adjusted with ``am_color``, ``pm_color``,
``am_marker``, and ``pm_marker``.

Temperature-corrected power view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``tc_power=True`` to view temperature-corrected power on the y-axis for
setups where raw power is used in the regression. This plotting option can be used
to review temperature-corrected power vs POA irradiance regardless of the presence
of a temperature-corrected power term in the regression. This an independent calculation
of temperature-corrected power for the plot.

.. code-block:: Python

    ct.scatter_plots(tc_power=True)

The layout can be controlled with ``tc_mode``:

- ``'replace'`` shows one plot with temperature-corrected power on the y-axis.
- ``'add_panel'`` shows raw power and temperature-corrected power in separate
  panels.
- ``'overlay'`` overlays raw power and temperature-corrected power in one plot.

.. code-block:: Python

    ct.scatter_plots(tc_power=True, tc_mode='add_panel')

.. note::

    The default temperature-corrected power calculation expects measured power,
    POA irradiance, and back-of-module temperature column groups. If a project
    uses different data, pass an explicit ``tc_power_calc`` dictionary that
    points to the correct columns or column groups.

    Passing a custom ``tc_power_calc`` dictionary can be used to calculate
    cell temperature from POA irradiance, ambient temperature, wind speed,
    module type, and mounting type. The dictionary must include a top-level
    ``power`` calculation tuple that produces the temperature-corrected power
    column, such as ``{'power': (power_temp_correct, {...})}``.

Linked timeseries
~~~~~~~~~~~~~~~~~
Use ``timeseries=True`` to add a timeseries panel below the scatter plot.
Selections in the scatter plot and timeseries plot are linked, which can help
identify when unusual scatter points occurred.

.. code-block:: Python

    ct.scatter_plots(timeseries=True)
    ct.scatter_plots(split_day=True, tc_power=True, tc_mode='overlay',
                     timeseries=True)

Adjusting reporting conditions
------------------------------
The selected ``test_setup`` provides default reporting-condition calculations,
but they can be adjusted for a specific project. For example, to use the 55th
percentile POA while leaving the other reporting-condition variables at their
default calculations:

.. code-block:: Python

    from captest.captest import perc_wrap

    ct.rep_cond(func={'poa': perc_wrap(55)})

The same adjustment can be saved in yaml:

.. code-block:: yaml

    captest:
      test_setup: e2848_default
      overrides:
        rep_conditions:
          func:
            poa: perc_55

The ``perc_55`` shorthand is converted to the corresponding percentile
function when the yaml file is loaded.

Custom setups
-------------
Most users should start with one of the built-in setups. If a project requires
a different regression equation, the setup can be customized by overriding the
regression formula and the measured/model column mappings.

See :ref:`custom_test_setups` for additional detail on the options.

For small changes to a built-in setup, use ``overrides``. For example, a yaml
file can change the reporting-condition calculation without redefining the
whole setup.

For a fully custom regression, use ``test_setup: custom`` and provide:

- ``reg_cols_meas``: how the measured data columns map to the regression terms.
- ``reg_cols_sim``: how the modeled data columns map to the regression terms.
- ``reg_fml``: the regression formula.

For example:

.. code-block:: yaml

    captest:
      test_setup: custom
      meas_path: ./data/measured/
      sim_path: ./data/pvsyst.csv
      ac_nameplate: 6_000_000
      overrides:
        reg_fml: power ~ poa + t_amb
        reg_cols_meas:
          power: real_pwr_mtr
          poa: irr_poa
          t_amb: temp_amb
        reg_cols_sim:
          power: E_Grid
          poa: GlobInc
          t_amb: T_Amb

.. note::

    Custom formulas and column mappings require more care than the built-in
    setups. Confirm that the measured and modeled columns use consistent units
    and represent the same physical quantities before comparing results.

.. _spec_corrected_poa:

Spectrally corrected POA (``e2848_spec_corrected_poa``)
-------------------------------------------------------
The ``e2848_spec_corrected_poa`` setup applies a First Solar spectral
correction to POA irradiance before running the standard ASTM E2848 regression.
This can be useful when spectral effects are part of the agreed test method.

The calculation uses pvlib's First Solar spectral-correction method, based on
the `McCarthy 2024 PVPMC poster
<https://pvpmc.sandia.gov/download/7822/?tmstv=1776198191>`_ and the
`pvlib.spectrum.spectral_factor_firstsolar
<https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.spectrum.spectral_factor_firstsolar.html>`_
reference.

Measured data requirements:

- A ``humidity`` column group with relative humidity in percent.
- A ``pressure`` column group with station pressure in hPa / mbar.
- A ``site`` dictionary on the measured ``CapData`` object. This is populated
  when :py:func:`~captest.io.load_data` is called with the ``site`` argument.

Modeled data requirements:

- A ``PrecWat`` column in the PVsyst output. Configure the PVsyst export to
  include precipitable water.

Example:

.. code-block:: Python

    from captest import CapTest, load_data, load_pvsyst

    site = {
        'loc': {'latitude': 33.01, 'longitude': -99.56,
                'altitude': 500, 'tz': 'America/Chicago'},
        'sys': {'surface_tilt': 0, 'surface_azimuth': 180, 'albedo': 0.2},
    }
    meas = load_data(path='./data/measured/', site=site)
    sim = load_pvsyst(path='./data/pvsyst_results.csv')

    ct = CapTest.from_params(
        test_setup='e2848_spec_corrected_poa',
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance='- 4',
        spectral_module_type='cdte',
    )

The corrected irradiance column is named ``poa_spec_corrected`` and is added
to both ``ct.meas.data`` and ``ct.sim.data`` during setup. The regression then
uses that corrected POA value in place of raw POA irradiance.

