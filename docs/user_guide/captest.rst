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
- Storing common test values, such as nameplate capacity, test tolerance,
  irradiance filter limits, shade-filter settings, and bifaciality.
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
:py:meth:`~captest.captest.CapTest.from_yaml` with both datasets present.

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

Running a capacity test
-----------------------
After ``ct`` has been created, use ``ct.meas`` and ``ct.sim`` in the same way
you would use separate ``CapData`` objects.

The example below shows the general pattern. Actual filters should be selected
to match the contract and test procedure; for the complete list of available
filtering methods, see the :ref:`Filtering section <capdata-api-filtering>` of
the CapData API reference.

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

    cap_ratio = ct.captest_results_check_pvalues()

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

Reviewing results
-----------------
The main comparison method is
:py:meth:`~captest.captest.CapTest.captest_results`. It predicts the measured
and modeled capacities at the reporting conditions, calculates the capacity
ratio, and can print a pass/fail summary using the AC nameplate and test
tolerance stored on your instance of Captest, e.g. ``ct``.

.. code-block:: Python

    cap_ratio = ct.captest_results()

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

