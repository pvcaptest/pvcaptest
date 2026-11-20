.. _pr_test:

Performance Ratio Test
======================
A performance ratio (PR) test compares the energy a plant delivered with the
energy it would have delivered at its DC nameplate under the irradiance (and,
optionally, the cell temperature) it actually saw. It is a different kind of
acceptance test from the ASTM E2848 capacity test described in :ref:`captest`:
there is no regression and there are no reporting conditions. The result is a
single measured ratio that is compared with an expected ratio and a tolerance.

pvcaptest runs PR tests through the same :py:class:`~captest.CapTest` object as
capacity tests. You pick a PR setup as ``test_setup``, apply filters to
``tst.meas`` (and ``tst.sim`` when the expected PR is derived from the model),
and call :py:meth:`~captest.CapTest.run_test` or
:py:meth:`~captest.CapTest.pr_results`. The whole test, filter pipelines and
validity checks included, round-trips through
:py:meth:`~captest.CapTest.to_yaml` like any other test.

The equations follow *Weather-Corrected Performance Ratio*, NREL/TP-5200-57991
(Dierauf et al., 2013). The measured PR is

.. math::

   PR = \frac{\sum E_{AC,i}}
             {\sum P_{nom}\,\frac{G_{POA,i}}{G_{STC}}\,
              \left[1 + \gamma\,(T_{cell,i} - T_{ref})\right] A_i\,\Delta t_i}

where :math:`\gamma` is ``power_temp_coeff / 100`` (per degree C),
:math:`A_i` is the availability factor, and the bracketed temperature term is
left out of a plain (not temperature-corrected) test.
The test passes when the measured PR is at least ``(1 - N / 100)`` times the
effective expected PR, where the tolerance is ``test_tolerance="- N"``.

Choosing a PR setup
-------------------
:py:func:`~captest.captest.pr_setups` lists the PR presets in
:data:`~captest.captest.TEST_SETUPS`; each is a
:py:class:`~captest.setup.PrSetup`. A PR test always names one of them (the
``"custom"`` setup is for regression tests only) and customizes it with
the key-by-key ``reg_cols_meas`` / ``reg_cols_sim`` overrides.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Preset
     - Flavor
   * - ``pr_plain``
     - No temperature correction; monofacial POA irradiance.
   * - ``pr_plain_bifi``
     - No temperature correction; total effective irradiance
       (:math:`E_{Total}`) for a bifacial plant.
   * - ``pr_tc_meas_bom``
     - Temperature corrected. Measured cell temperature is calculated from the
       measured back-of-module temperature; the modeled cell temperature is the
       PVsyst ``TArray`` output. A contract variant, not the NREL protocol.
   * - ``pr_tc_meas_bom_bifi``
     - As ``pr_tc_meas_bom`` with total effective irradiance.
   * - ``pr_tc_calc_bom``
     - Temperature corrected following the NREL protocol: back-of-module and
       cell temperature are calculated with the Sandia model from POA, ambient
       temperature and wind speed on both the measured and modeled sides.
   * - ``pr_tc_calc_bom_bifi``
     - As ``pr_tc_calc_bom`` with total effective irradiance.

Read any preset's ``description`` programmatically, for example
``captest.TEST_SETUPS["pr_tc_meas_bom"].description``.

The bifacial presets, like the ``*_rear_shade_sim`` regression presets, carry
rear-shading losses in the modeled ``rpoa_pvsyst`` and so require
:py:attr:`~captest.CapTest.rear_shade` to stay at ``0``.

A first test
------------
The example below is a substantial-completion style test: temperature
corrected, reference derived from a PVsyst model, tolerance of 5 % of the
expected PR.

.. code-block:: Python

   import captest as ct

   tst = ct.CapTest.from_params(
       test_setup="pr_tc_meas_bom",
       meas_path="./data/scada.csv",
       sim_path="./data/pvsyst_hourly.csv",
       dc_nameplate=12_500_000,        # W
       power_temp_coeff=-0.35,         # %/degC, negative for silicon
       test_tolerance="- 5",
   )

   tst.sim.filter_time(days=30, test_date="2025-06-12")
   tst.sim.filter_irr(200, 2000, col_name="GlobInc")
   tst.meas.filter_irr(300, 1200, col_name="irr_poa_mean_agg")

   results = tst.run_test()
   print(results)

``run_test`` runs :py:meth:`~captest.CapTest.setup`, replays the filters you
applied on each side, derives the reference, computes the measured PR, evaluates
any validity checks and returns a :py:class:`~captest.prtest.PrTestResults`.
:py:meth:`~captest.CapTest.pr_results` does the calculation alone, on the
already set up and filtered data; use it when you set up and filtered
interactively.

``results.ratio`` is the measured PR, ``results.expected_ratio`` the effective
expected PR, ``results.threshold`` the pass line and ``results.passed`` the
verdict. ``results.valid`` reports the validity checks separately from
``passed``. ``results.intervals`` holds the per-interval energy, expected
energy, irradiance, cell temperature and PR for plotting or export. The
headline accessors ``kind``, ``ratio``, ``expected_ratio`` and ``passed`` are
shared with :py:class:`~captest.captest.CapTestResults` (whose ``kind`` is
``"cap_ratio"``; a PR result's is ``"pr"``).

:py:attr:`~captest.CapTest.test_kind` tells the two kinds apart (``"pr"`` or
``"regression"``).

.. note::

   Reporting conditions do not apply to a PR test. :py:meth:`~captest.CapTest.rep_cond`
   raises, and a ``RepCond`` step, or an ``Irradiance`` step whose ``ref_val``
   is ``"rep_irr"`` (or its legacy alias ``"self_val"``), in a PR filter
   pipeline is rejected before replay with a ``ValueError`` naming the side and
   step. A PR run never computes or writes reporting conditions.

.. _pr-reference:

The expected PR and reference temperature
-----------------------------------------
The expected PR (and, on a temperature-corrected test, the reference
temperature :math:`T_{ref}`) come from one of two sources, chosen together by
:py:attr:`~captest.CapTest.reference_source`. The two are never mixed.

``"derived"`` (the default)
   Both are computed from the modeled data after the ``tst.sim`` filter
   pipeline runs. The expected PR is the plain PR of the kept modeled rows,
   and :math:`T_{ref}` is their irradiance-weighted cell temperature. With
   these two values the plain and temperature-corrected PRs of the modeled data
   agree exactly, which is what makes the derived expected PR a fair target for
   the corrected measured PR. A modeled side is required.

``"fixed"``
   Both are stated numbers, for example from a contract:
   ``expected_pr`` and, on a temperature-corrected test, ``t_ref``. No modeled
   side is needed, so ``sim`` can be left out entirely. Because
   ``base_temp`` defaults to 25, a fixed temperature-corrected test must set
   ``t_ref`` explicitly so it never corrects to STC by accident.

.. code-block:: Python

   tst = ct.CapTest(
       meas=meas,
       test_setup="pr_tc_meas_bom",
       reference_source="fixed",
       expected_pr=0.835,
       t_ref=42.0,
       dc_nameplate=12_500_000,
       power_temp_coeff=-0.35,
       test_tolerance="- 3",
   )
   tst.setup()

Single or monthly values
~~~~~~~~~~~~~~~~~~~~~~~~
:py:attr:`~captest.CapTest.reference_granularity` is ``"single"`` (one value
for the whole test, the default) or ``"monthly"`` (one per calendar month).
Monthly references work in both reference modes:

- Fixed: give ``expected_pr_monthly`` and, when temperature corrected,
  ``t_ref_monthly``, each a list of 12 values, January first. A test month
  missing from a table it needs raises, and a ``t_ref_monthly`` on a plain test
  raises rather than being ignored.
- Derived: the kept modeled rows are grouped by calendar month (the month
  number only; the year of a PVsyst typical year is ignored) and each month
  gets its own expected PR and :math:`T_{ref}`. A test month with no kept modeled
  rows raises, and a month that is only partly present (a run of seven or more
  consecutive days with no kept rows, as a ``Time`` filter leaves) warns,
  because a partial month is not the full monthly value.

.. code-block:: Python

   tst = ct.CapTest(
       meas=meas,
       test_setup="pr_tc_meas_bom",
       reference_source="fixed",
       reference_granularity="monthly",
       expected_pr_monthly=[0.82, 0.84, 0.85, 0.86, 0.86, 0.85,
                            0.84, 0.84, 0.85, 0.86, 0.84, 0.82],
       t_ref_monthly=[38, 41, 44, 47, 50, 53, 55, 54, 50, 46, 42, 39],
       dc_nameplate=12_500_000,
       power_temp_coeff=-0.35,
       test_tolerance="- 3",
   )

With monthly values the two quantities are used differently, and it matters:

- The *measured* PR is one ratio over the whole test. Each interval's expected
  energy uses the :math:`T_{ref}` of **its own month**, so a test that spans a
  month boundary mixes two reference temperatures inside one sum.
  :math:`T_{ref}` is therefore never combined across months; the
  ``results.intervals["t_ref"]`` column records the value applied to every row,
  and ``results.t_ref`` is ``None`` for a monthly test.
- The *expected* PR is combined at the PR level, weighting each month's value
  by the measured POA insolation of that month over the kept rows:
  :math:`PR_{exp} = \sum_m H_m\,EPR_m / \sum_m H_m`. A partial month weighs only
  the days inside the test.

.. note::

   Weighting by insolation, as contracts that state monthly values write it,
   is an approximation of what the measured PR does implicitly (weighting each
   month by its expected energy). The two differ by each month's average
   temperature correction and availability, a second-order effect.
   ``results.monthly`` lists, per month, the kept intervals, insolation,
   expected PR, :math:`T_{ref}` and that month's own measured PR.

Energy or power numerators
--------------------------
The measured numerator can be interval power (the default) or interval energy.
Meter power is a common substitute when the meter's accumulated energy is
unreliable. Switching is a configuration change: name the numerator and its
units per side with ``meas_numerator_kind`` / ``meas_numerator_units`` (and
``sim_numerator_kind`` / ``sim_numerator_units`` for the modeled side).
Units are ``W`` / ``kW`` / ``MW`` for power and ``Wh`` / ``kWh`` / ``MWh`` for
energy. A power numerator is converted to energy with the timestep of the side's
*unfiltered* index, so filtering never changes it (an energy numerator, where
the timestep matters most, would otherwise be off by the thinning factor).

A PR setup names its inputs as *roles*: ``power``, ``energy``, ``poa``,
``cell_temp``, ``front_poa`` and ``availability``. Filters and plots keep
finding ``power`` and ``poa`` just as on a capacity test. The ``power`` role
always holds physical power, because power filters such as
:py:meth:`~captest.capdata.CapData.filter_power` compare it with power
thresholds. So, for an energy numerator:

- Give the ``energy`` role and remove the preset's ``power`` role (``None``
  removes a term). The ``power`` role is then derived as energy divided by the
  timestep with the registered ``energy_to_power`` calculation
  (:py:func:`~captest.calcparams.energy_to_power`), expressed in
  ``meas_power_units``, so a clipping threshold means the same thing whichever
  numerator you use.
- If you supply both roles (meter energy for the numerator, inverter power for
  clipping filters), nothing is derived and the ``power`` role is taken to be in
  ``meas_power_units``.
- On a **power** numerator the numerator unit and the power unit are the same
  declaration, so ``meas_numerator_units`` must equal ``meas_power_units``; they
  may differ only for an energy numerator.

.. code-block:: Python

   tst = ct.CapTest(
       meas=meas,
       test_setup="pr_plain",
       reference_source="fixed",
       expected_pr=0.81,
       dc_nameplate=12_500_000,
       test_tolerance="- 5",
       reg_cols_meas={
           "energy": {"column": "meter_kwh"},
           "power": None,
       },
       meas_numerator_kind="energy",
       meas_numerator_units="kWh",
       meas_power_units="kW",
   )
   tst.setup()

.. _pr-registers:

Cumulative energy registers
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Meter energy often arrives as a cumulative register. Turn it into interval
energy with the registered calculation
:py:func:`~captest.calcparams.register_to_interval_energy`, used as the
``energy`` node. Nodes evaluate on ``tst.meas.data``, so the difference is
taken before any filter runs.

.. code-block:: Python

   reg_cols_meas={
       "energy": {
           "calc": "register_to_interval_energy",
           "args": {"register": "meter_kwh_total", "label": "start"},
       },
       "power": None,
   }

The calculation guards against three silent failures; each gives NaN, which
the PR drops from every sum and counts:

- an interval that is not exactly one timestep long (a missing row, so the
  difference would span several intervals);
- a negative difference (a register reset or rollover), with a warning;
- a NaN reading at either end.

``label`` says where the value is stamped. ``register[t + dt] - register[t]``
(``label="start"``) matches interval-average power and irradiance stamped at the
interval start, which is the pandas resample default. Use ``label="end"`` when
the timestamps mark the end of the interval. Whichever you choose should match
the convention of the irradiance, and it also decides which month an interval at
a month boundary belongs to.

If the data must be coarser than the register's native resolution, difference
at the **native** resolution first and then sum the interval energy into bins
with :py:func:`~captest.prtest.resample_interval_energy`, using the same
``label`` and ``closed`` settings as the irradiance resampling, so energy and
irradiance cover the same bins:

.. code-block:: Python

   from captest import calcparams, prtest

   energy = calcparams.register_to_interval_energy(
       raw, "meter_kwh_total", label="start", verbose=False
   )
   energy_15min = prtest.resample_interval_energy(energy, "15min")

A bin is valid only when every native interval in it is present and valid. One
missing reading, a reset inside the bin or an incomplete first or last bin gives
NaN for the whole bin, not a partial sum that quietly understates the energy
(pandas' plain ``sum`` of a bin missing one of fifteen one-minute intervals
returns 14/15 of the energy). Do not resample the register itself with
``resample(...).last()`` before differencing: with pandas' default left label
that puts each bin's final reading on its start timestamp and misaligns the
energy.

A resample prep step that carries this completeness rule is planned. Until it
lands, resample before loading into ``CapData`` or in a custom loader.

Front POA on bifacial tests
---------------------------
On a bifacial PR test the ``poa`` role is the total effective irradiance, which
is what the expected-energy denominator and the :math:`T_{ref}` weights use.
Contract insolation requirements, however, are usually written on front POA, so
the bifacial presets also name a ``front_poa`` role (the front POA aggregate;
it defaults to the ``poa`` node when absent). The checks read ``front_poa`` by
default. The temperature chain (back-of-module and cell temperature) also uses
front POA, since the heat-transfer coefficients are front-side quantities.

:py:meth:`~captest.capdata.CapData.filter_irr` resolves the ``poa`` role, which is
:math:`E_{Total}` on a bifacial test. To filter on front POA instead, pass the
column explicitly, looking it up from the role mapping:

.. code-block:: Python

   front = tst.meas.regression_cols["front_poa"]
   tst.meas.filter_irr(300, 1200, col_name=front)

Availability
------------
Availability, an optional measured-side input, scales the expected energy:
``expected * A``. Give it either as a scalar with
:py:attr:`~captest.CapTest.availability` (default ``1``) or per interval as an
``availability`` role, for example
``reg_cols_meas={"availability": {"column": "avail"}}``; do not set both. Values
must lie between 0 and 1 (a percent series of 0 to 100 raises rather than
inflating the denominator a hundredfold).

Availability applies to the measured side only. The modeled reference always
uses ``A = 1``, and an ``availability`` role on the modeled side is rejected.

.. warning::

   Use availability as a factor **or** filter out the unavailable intervals,
   never both. Filtering them out and applying the factor counts the loss
   twice.

The results report the measured PR with the factor (``results.ratio``) and
without it (``results.ratio_without_availability``), since some contracts judge
a case on the PR without the availability adjustment.

Degradation
-----------
Set ``degradation_rate`` (percent per year; 0.4 means 0.4 %/yr) and
``degradation_years`` together. The effective expected PR is the expected PR
times ``1 - degradation_years * degradation_rate / 100``, a linear derate: a rate
of 0.4 over 1.5 years gives a factor of 0.994. The undegraded value stays on the
result as ``expected_ratio_undegraded`` and the rate, years and factor in
``results.degradation``. A factor of zero or below raises.
:py:func:`~captest.prtest.degradation_years` computes the average age over a
test period from a warranty start date.

The derate is applied to the **reference** side so that the measured PR stays the
plain measured value a contract reports. For pass/fail this is equivalent to
derating the measured side's expected energy by the same factor:
``PR_m / (1 - y*d) >= t * PR_exp`` exactly when
``PR_m >= t * PR_exp * (1 - y*d)``, and the equivalence holds as well for the
ratio ``PR_m / threshold``. It does not hold for an absolute margin
(``results.margin`` is defined against the derated threshold).

Temperature settings
--------------------
**Sign of the coefficient.** ``power_temp_coeff`` is in %/degC and **negative**
for crystalline silicon (for example ``-0.35``). A positive value on a
temperature-corrected test raises, because it would move the correction the
wrong way. A contract that states the coefficient unsigned in a form such as
``1 + β·(Tref − Tmeas)`` uses the same equation as pvcaptest's
``1 + γ·(Tcell − Tref)`` with the sign folded in: an unsigned β of 0.32 is
``power_temp_coeff=-0.32``. A warning is given when the magnitude is below 0.05
or above 1 %/degC, which usually means a fraction was entered as a percent.

**Modeled cell temperature.** The ``pr_tc_meas_bom`` presets take the modeled
cell temperature from the PVsyst ``TArray`` output. The ``pr_tc_calc_bom``
presets calculate it with the Sandia chain, from ``GlobInc``, ``T_Amb`` and
``WindVel``, on both sides. Overriding one for the other is a single
``reg_cols_sim`` key. The NREL paper stresses that using the same heat-transfer
model for :math:`T_{ref}` and the measured cell temperature is the stricter
choice. This is not enforced.

**Cell-to-back-of-module offset.** The conduction offset in
:math:`T_{cell} = T_{BOM} + (G/1000)\,\Delta T` is taken from the Sandia
coefficients for the module and racking type. Override it with the ``temp_delta``
argument of the ``cell_temp`` node (for instance ``0.5`` for an insulated sensor):

.. code-block:: Python

   reg_cols_meas={
       "cell_temp": {
           "calc": "cell_temp",
           "args": {
               "bom": {"group": "temp_bom", "agg": "mean"},
               "poa": {"group": "irr_poa", "agg": "mean"},
               "temp_delta": 0.5,
           },
       },
   }

**Hourly model, sub-hourly measurements.** A PVsyst run is usually hourly and
the measured data sub-hourly. The averaging period differs between the two sides,
which the NREL paper notes has a small effect on cell temperature. Each side uses
its own timestep, taken from its unfiltered index.

Validity checks
---------------
Validity checks state requirements the data must meet, such as a minimum number
of high-irradiance intervals. A check reports a
:py:class:`~captest.checks.CheckResult` and **never removes rows**. The
``check_*`` methods of :py:class:`~captest.capdata.CapData` register a check
without evaluating it; the PR run evaluates them and puts the results in
``results.checks`` (keyed ``"meas: <label>"`` / ``"sim: <label>"``) and the
overall ``results.valid``. ``valid`` is independent of ``passed``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Requirement
   * - :py:meth:`~captest.capdata.CapData.check_interval_count`
     - At least N intervals at or above an irradiance threshold.
   * - :py:meth:`~captest.capdata.CapData.check_duration`
     - At least H hours (intervals times the timestep) at or above a
       threshold, not necessarily continuous.
   * - :py:meth:`~captest.capdata.CapData.check_continuous_hours`
     - At least N days holding a continuous run of at least H hours above a
       threshold.
   * - :py:meth:`~captest.capdata.CapData.check_daily_insolation`
     - At least N days with daily insolation above a threshold.
   * - :py:meth:`~captest.capdata.CapData.check_nrel_irradiance`
     - The NREL minimum-irradiance criteria (see below).
   * - :py:meth:`~captest.capdata.CapData.check_any_of`
     - Passes when any member passes, so tiered contract schemes are data.
   * - :py:meth:`~captest.capdata.CapData.check_custom`
     - Calls your function on the selected rows.

Which rows a check reads is set by its ``data`` argument. Count and duration
checks default to the **PR rows** (the filtered rows with every PR input
present), so a check can only certify intervals that enter the PR. Day-based
checks (continuous hours, daily insolation, NREL) default to the **unfiltered**
data inside the test period, because removed rows would break a continuous run
and understate daily insolation by construction. The period defaults to the first
through last calendar day holding a kept measured row; give ``start`` and ``end``
for an explicit one. "Continuous" means consecutive timestamps on the timestep
grid, so a NaN irradiance or a missing timestamp ends a run. Irradiance checks
read the ``front_poa`` role by default; the ``irr`` argument names another role
or a column.

.. code-block:: Python

   tst.meas.check_duration(400, 30, custom_name="30 h above 400")
   tst.meas.check_any_of(
       tst.meas.check_interval_count(900, 40, custom_name="high tier"),
       tst.meas.check_interval_count(400, 120, custom_name="low tier"),
   )

   results = tst.run_test()
   results.valid
   results.checks["meas: AnyOf"].detail

A custom check takes a function of the selected frame that returns a bool, or a
``(bool, DataFrame)`` tuple whose frame becomes the detail table:

.. code-block:: Python

   def enough_valid_data(frame, minimum_fraction):
       return frame["irr_poa_mean_agg"].notna().mean() >= minimum_fraction

   tst.meas.check_custom(
       enough_valid_data, minimum_fraction=0.99, custom_name="at least 99 % valid"
   )

Define the function in a module (not a lambda or a function local to a notebook
cell) when the test is to be saved: a check is written to the config by the
function's ``module:qualname``.

.. warning::

   Loading a config that contains a custom check imports and calls the
   function it names. Only load configs from trusted sources.

**The NREL reading.** The protocol's minimum-irradiance passage is not internally
consistent ("at least three days..." and later "five days that meet these
criteria"). pvcaptest reads it as counts over the test period: at least five days
with daily insolation of 3000 Wh/m² or more, and among those at least three with
3 continuous hours above 600 W/m². On a period of exactly five days this is the
reading that every one of the five days reaches the insolation and three of them
the continuous run. The thresholds, day counts and hours are parameters.

Checks never extend the test period or search for the strongest days; those are
decisions for the test engineer, and the check results state the facts they
rest on.

Saving and reloading
--------------------
:py:meth:`~captest.CapTest.to_yaml` and :py:meth:`~captest.CapTest.to_mapping`
write a PR test as one config: the PR parameters above, the ``overrides`` that
differ from the preset, and the ``meas_filters`` / ``sim_filters`` pipelines.
They also write the validity checks registered on each side
(``meas_checks`` / ``sim_checks``), for PR tests only. A PR mapping omits the
regression-only keys (``reg_fml``, ``rep_conditions``, ``rc_source`` and the
reporting-condition values), and a regression mapping never contains a PR key.

.. note::

   Validity checks are saved for PR tests only. A regression test with checks
   registered on ``tst.meas`` or ``tst.sim`` warns that they are not saved;
   register them again after loading, or round-trip them with
   :py:meth:`~captest.capdata.CapData.checks_to_config` and
   :py:meth:`~captest.capdata.CapData.set_checks`.

.. code-block:: Python

   tst.to_yaml("pr_test.yaml")
   reloaded = ct.CapTest.from_yaml("pr_test.yaml")
   results = reloaded.run_test()

See :ref:`saving_reproducing` for loaders and the other things a config does
and does not carry.

The standalone performance ratio
--------------------------------
:py:meth:`CapData.performance_ratio <captest.capdata.CapData.performance_ratio>`
computes the measured PR of one dataset, with no PVsyst model, no contract and no
pass/fail. It takes column names and acts on the filtered data, with the timestep
taken from the unfiltered index. :py:func:`captest.prtest.performance_ratio` does
the same for a DataFrame.

.. code-block:: Python

   # plain PR
   results = cd.performance_ratio(
       numerator="meter_kw", poa="irr_poa_mean_agg", dc_nameplate=12_500_000,
       kind="power", units="kW",
   )

   # temperature corrected: a cell temperature column, a back-of-module
   # column, or ambient temperature with wind speed
   results = cd.performance_ratio(
       numerator="meter_kw", poa="irr_poa_mean_agg", dc_nameplate=12_500_000,
       kind="power", units="kW",
       bom="temp_bom_mean_agg", power_temp_coeff=-0.35, base_temp=45.0,
   )
   results.ratio

The result is a :py:class:`~captest.prtest.PrTestResults` whose ``passed``,
``valid`` and ``expected_ratio`` are ``None``. A temperature source selects the
corrected form with ``base_temp`` as the reference temperature (default 25 degC).
There is deliberately no option to correct to the irradiance-weighted mean
temperature of the data itself: by the identity above that returns exactly the
plain PR.

Migrating from ``perf_ratio``
-----------------------------
``prtest.perf_ratio``, ``prtest.perf_ratio_temp_corr_nrel``,
``prtest.perf_ratio_inputs_ok`` and ``PrResults`` were removed without
deprecation shims.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Old
     - New
   * - ``perf_ratio(ac_energy, dc_nameplate, poa)``
     - ``cd.performance_ratio(numerator=..., poa=..., dc_nameplate=...,
       kind=..., units=...)``, or ``prtest.performance_ratio(df, ...)`` on a
       DataFrame. Pass column names, not Series.
   * - ``unit_adj=1000``
     - ``kind="power"`` or ``"energy"`` plus ``units`` (``W`` / ``kW`` / ``MW``,
       ``Wh`` / ``kWh`` / ``MWh``).
   * - ``perf_ratio_temp_corr_nrel(..., temp_bom=, temp_amb=, wind_speed=)``
     - The same call with ``cell_temp=``, ``bom=``, or ``temp_amb=`` with
       ``wind_speed=``, and ``power_temp_coeff=`` (%/degC, negative).
   * - ``base_temp=25``
     - ``base_temp=``, only with a temperature source.
   * - ``availability=``
     - ``prtest.measured_pr(..., availability=...)``. In a CapTest PR test, the
       ``availability`` parameter or role.
   * - ``degradation=``, ``year=`` (compounded ``(1 - d) ** year``)
     - ``degradation_rate`` (%/yr) and ``degradation_years`` on
       :py:class:`~captest.CapTest`, or ``prtest.degradation_factor(rate, years)``
       applied to the expected PR. The derate is now linear,
       ``1 - years * rate / 100``.
   * - ``PrResults``
     - :py:class:`~captest.prtest.PrTestResults`.
   * - A PR test assembled in a notebook
     - A PR preset and :py:meth:`~captest.CapTest.run_test`, as above.

The standalone ``performance_ratio`` takes no availability. To apply it outside
a CapTest, build the complete-row set with
:py:func:`~captest.prtest.complete_rows` over every input, availability
included, and call :py:func:`~captest.prtest.measured_pr` on those rows.
