.. _custom_test_setups:

Custom Test Setups
==================
The built-in ``test_setup`` presets (``e2848_default``,
``bifi_e2848_etotal_rear_shade_sim``, ``bifi_e2848_etotal_rear_shade_meas``,
``bifi_power_tc_meas_tbom``, ``bifi_power_tc_calc_tbom``, ``e2848_spec_corrected_poa``,
and their variants) cover the most common capacity-test configurations. When a
project calls for a different regression equation, different sensors, or a
non-standard column calculation, pvcaptest lets you change one term of a preset,
or write a complete setup of your own, without modifying the package.

A test setup is a **document**: plain yaml (or json, or a Python dict) made only of
strings, numbers, booleans, lists and tagged mappings. Every calculation is named,
never held as a Python function, so a setup can be saved to a file, compared, and
reloaded exactly. The built-in presets are documents too; each one ships as
``captest/setups/<name>.yaml`` and is loaded into
:data:`~captest.captest.TEST_SETUPS` as a :py:class:`~captest.setup.TestSetup`.

This page explains the node grammar of the ``reg_cols_meas`` / ``reg_cols_sim``
mappings, shows how the :doc:`/source/api_reference/calcparams` calculations and
your own registered calculations plug into them, and describes the ways to wire a
custom setup into a :py:class:`~captest.CapTest`.

The regression column grammar
-----------------------------
Each key in ``reg_cols_meas`` or ``reg_cols_sim`` is a variable of the regression
formula (such as ``power`` or ``poa``) and maps to a **node**. A node is a mapping
with exactly one of the keys ``group``, ``column`` or ``calc``, which says what
kind of node it is:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Node
     - Meaning
   * - ``{group: irr_poa, agg: mean}``
     - Aggregate the columns of a ``column_groups`` group with
       ``CapData.agg_group``; writes the column ``irr_poa_mean_agg``. ``agg`` is
       one of ``mean``, ``sum``, ``median``, ``min``, ``max`` and defaults to
       ``mean``.
   * - ``{column: E_Grid}``
     - Use one column of ``data`` by name, as is.
   * - ``{calc: e_total, args: {...}}``
     - Run a registered calculation. ``args`` are its keyword arguments, each
       value a node or a literal. It writes a column named for the calculation
       (``e_total``), which is what the parent node or the regression uses.
   * - ``"cdte"``, ``3.5``, ``true``, ``null``, ``[1, 2]``
     - A literal argument, passed to the calculation unchanged. Allowed only as a
       value inside ``args``.

For example, the measured side of a preset that regresses power against front POA
irradiance aggregates the POA sensors and the power meters:

.. code-block:: yaml

    reg_cols:
      power: {group: real_pwr_mtr, agg: sum}
      poa: {group: irr_poa, agg: mean}

This example assumes that you have a key in your ``CapData.column_groups`` called
``irr_poa`` that points to a list of columns, which contain measurements from POA
irradiance sensors. See the :ref:`Column Grouping <col-grouping>` section of the
CapData workflow documentation page for additional explanation. The modeled side of
the built-in setups reads PVsyst output columns directly:

.. code-block:: yaml

    reg_cols:
      power: {column: E_Grid}
      poa: {column: GlobInc}

A calculation node names a registered calculation and gives its arguments by
keyword. Nodes nest to any depth:

.. code-block:: yaml

    poa:
      calc: e_total
      args:
        poa: {group: irr_poa, agg: mean}
        rpoa: {group: irr_rpoa, agg: mean}

During :py:meth:`~captest.CapTest.setup` (or
:py:meth:`~captest.capdata.CapData.process_regression_columns`), pvcaptest
evaluates the tree bottom-up: each ``group`` node aggregates its sensors, each
``calc`` node adds a column named for the calculation to ``CapData.data``, and that
column name is passed upward as the argument of the parent node. For the example
above, pvcaptest adds ``irr_poa_mean_agg``, ``irr_rpoa_mean_agg`` and ``e_total``
columns to ``data``, and ``regression_cols['poa']`` becomes ``'e_total'``.

The rules of the grammar:

- **No shorthand.** A bare string is always a literal, never a column or a group.
  Write ``{column: GlobInc}`` or ``{group: irr_poa}``. A formula variable must map
  to a ``group``, ``column`` or ``calc`` node; a literal there is an error.
- **Keyword-only arguments.** ``args`` bind by name; the order of the function's
  parameters is never used. Every required argument must be present, and an
  argument the function does not take is an error.
- **Literals.** Strings, finite numbers, booleans, ``null`` and lists of those.
  ``null`` reaches the function as ``None``. A mapping without one of the three
  node keys is an error, not a literal.
- **One column per producer.** Two nodes on one side may write the same column
  only if they are the same node. Using one calculation twice with different
  ``args`` on the same side is an error, because both would write the same column.

Setups are validated when they are loaded, before any data is read, and every
error names the path in the document where it occurred (for example
``meas.reg_cols.power.args.cell_temp.calc``).

.. note::

    Aggregation uses `pandas.DataFrame.agg <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html>`_.
    A setup document accepts the ``agg`` names listed above.

Using calcparams functions
--------------------------
The functions in :doc:`/source/api_reference/calcparams` are registered under
their own names in :data:`~captest.calcparams.CALC_REGISTRY`, and a ``calc`` node
refers to them by that name. There is nothing to import:

.. code-block:: yaml

    cell_temp:
      calc: cell_temp
      args:
        poa: {group: irr_poa, agg: mean}
        bom: {group: temp_bom, agg: mean}

To list the calculation names available in a session:

.. code-block:: Python

    from captest.calcparams import CALC_REGISTRY

    sorted(CALC_REGISTRY)

Using custom functions
----------------------
If you need to calculate a parameter that has no function in the ``calcparams``
module, write your own and register it with
:py:func:`~captest.calcparams.register_calc`. The function follows the same
conventions as the built-in calculations:

- The first argument is ``data``, the source DataFrame.
- Other arguments are keyword arguments. An argument filled by a node receives
  the name of the column that node produced.
- It returns a :class:`pandas.Series` indexed like ``data``.

.. code-block:: Python

    from captest.calcparams import register_calc

    @register_calc()
    def my_adjusted_poa(data, poa=None, adjustment=1.0, verbose=True):
        """Scale a POA column by a site-specific adjustment factor."""
        if verbose:
            print(f"Calculating my_adjusted_poa as {poa} * {adjustment}")
        return data[poa] * adjustment

A setup then names it like any other calculation:

.. code-block:: yaml

    poa:
      calc: my_adjusted_poa
      args:
        poa: {group: irr_poa, agg: mean}
        adjustment: 1.05

The function must be registered in your code **before** the setup that names it
is loaded, for example in the notebook or script that builds the ``CapTest``.
A setup document stores only the name; loading one never imports code. A name
that is not registered is an error with a "did you mean" hint. Running the
defining cell again replaces the registered function (a redefinition has the
same module and qualified name), while registering a *different* function under a
name that is already taken raises ``ValueError``.

The registry name defaults to the function's ``__name__`` and is also the name of
the column the calculation writes; pass ``register_calc(name=...)`` to choose a
different one. If the calculation uses one of the ``CapTest`` scalars described
below, list it in ``requires_params`` so a missing value is reported before the
test runs, and list any optional package it imports in ``requires_import``:

.. code-block:: Python

    @register_calc(requires_params=("power_temp_coeff",))
    def power_temp_correct_clipped(
        data, power, cell_temp, power_temp_coeff=None, cap=0.98, verbose=True
    ):
        ...

Adding a verbose kwarg to print an explanation of the calculation is not required, but
strongly recommended as it makes the calculation traceable for a reviewing party.

Creating a Custom Regression Columns Dictionary
-----------------------------------------------
The example below writes measured and modeled regression columns that compute
temperature-corrected power from raw power, back-of-module temperature
(estimated from POA, ambient temperature, and wind speed), and cell
temperature. As yaml, the complete setup document is:

.. code-block:: yaml

    name: power_tc_poa
    description: Temperature-corrected power against front POA irradiance.
    reg_fml: power ~ poa
    meas:
      reg_cols:
        power:
          calc: power_temp_correct
          args:
            power: {group: real_pwr_mtr, agg: sum}
            cell_temp:
              calc: cell_temp
              args:
                poa: {group: irr_poa, agg: mean}
                bom:
                  calc: bom_temp
                  args:
                    poa: {group: irr_poa, agg: mean}
                    temp_amb: {group: temp_amb, agg: mean}
                    wind_speed: {group: wind_speed, agg: mean}
        poa: {group: irr_poa, agg: mean}
    sim:
      reg_cols:
        power:
          calc: power_temp_correct
          args:
            power: {column: E_Grid}
            cell_temp: {column: TArray}
        poa: {column: GlobInc}
    rep_conditions:
      func: {poa: perc_60}

The same regression columns in Python are plain dicts with the same shape:

.. code-block:: Python

    my_meas_cols = {
        "power": {
            "calc": "power_temp_correct",
            "args": {
                "power": {"group": "real_pwr_mtr", "agg": "sum"},
                "cell_temp": {
                    "calc": "cell_temp",
                    "args": {
                        "poa": {"group": "irr_poa", "agg": "mean"},
                        "bom": {
                            "calc": "bom_temp",
                            "args": {
                                "poa": {"group": "irr_poa", "agg": "mean"},
                                "temp_amb": {"group": "temp_amb", "agg": "mean"},
                                "wind_speed": {"group": "wind_speed", "agg": "mean"},
                            },
                        },
                    },
                },
            },
        },
        "poa": {"group": "irr_poa", "agg": "mean"},
    }

    my_sim_cols = {
        "power": {
            "calc": "power_temp_correct",
            "args": {"power": {"column": "E_Grid"}, "cell_temp": {"column": "TArray"}},
        },
        "poa": {"column": "GlobInc"},
    }

A setup document saved to a file can be loaded and validated on its own with
:py:meth:`TestSetup.load <captest.setup.TestSetup.load>`, which accepts a file
path or a mapping (:py:meth:`TestSetup.loads <captest.setup.TestSetup.loads>`
accepts document text):

.. code-block:: Python

    from captest.setup import TestSetup

    power_tc_poa = TestSetup.load("./power_tc_poa.yaml")
    power_tc_poa.content_digest()   # stable identity of the normalised document

Scalar auto-injection
---------------------
Scalar parameters such as ``power_temp_coeff``, ``base_temp``,
``bifaciality``, and ``spectral_module_type`` do not need to appear in the
setup. When a calculation has a keyword argument whose name matches an attribute
on the ``CapData`` instance, and the node's ``args`` do not give that argument,
pvcaptest injects the attribute's value. :py:class:`~captest.CapTest` propagates
these scalars onto the ``CapData`` instances during
:py:meth:`~captest.CapTest.setup`, so setting them on the
:py:class:`~captest.CapTest` instance is sufficient:

.. code-block:: Python

    tst = CapTest.from_params(
        test_setup="custom",
        reg_cols_meas=my_meas_cols,
        reg_cols_sim=my_sim_cols,
        reg_fml="power ~ poa",
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        power_temp_coeff=-0.36,   # injected automatically into power_temp_correct
        base_temp=25,             # injected automatically into power_temp_correct
    )

Each argument of a calculation is resolved in this order:

1. the node's ``args`` entry, when the key is present, including an explicit
   ``null``, which reaches the function as ``None``;
2. otherwise the ``CapData`` attribute of that name, when it exists and is not
   ``None``;
3. otherwise the function's own default.

So to override the injected value for one calculation, give it in that node's
``args`` (for example ``args: {..., base_temp: 20}``).

This approach is recommended because it ensures values that should be consistent between
the measured data and simulated data, like ``bifaciality``, match.

A setup can also *require* a value with its ``params`` field. The built-in
``*_rear_shade_sim`` presets declare ``params: {rear_shade: 0}`` because the
modeled rear irradiance already carries the rear shading; with one of them,
``setup()`` refuses a non-zero ``rear_shade``. Override ``params`` like any other
setup field.

Wiring a custom setup into CapTest
----------------------------------
There are three ways to supply custom regression columns.

**Route 1 — fully custom setup.** Pass ``test_setup='custom'`` with the complete
``reg_cols_meas``, ``reg_cols_sim`` and ``reg_fml``. Every formula variable must be
present on both sides, because there is no preset to fill anything in.
``scatter_plots`` defaults to ``default`` and the reporting conditions to the mean
of every variable if omitted:

.. code-block:: Python

    from captest import CapTest

    tst = CapTest.from_params(
        test_setup="custom",
        reg_cols_meas=my_meas_cols,
        reg_cols_sim=my_sim_cols,
        reg_fml="power ~ poa",
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance="- 4",
    )

In a yaml config file the same setup is written under ``overrides``:

.. code-block:: yaml

    captest:
      test_setup: custom
      ac_nameplate: 6_000_000
      overrides:
        reg_fml: power ~ poa
        reg_cols_meas:
          power: {group: real_pwr_mtr, agg: sum}
          poa: {group: irr_poa, agg: mean}
        reg_cols_sim:
          power: {column: E_Grid}
          poa: {column: GlobInc}

**Route 2 — override terms of a named preset.** If a built-in preset is right
apart from one or two terms, pass ``reg_cols_meas`` and / or ``reg_cols_sim``
alongside the named ``test_setup``. The overrides are merged **key by key** onto
the preset: a key you give replaces that variable's whole node, every key you
leave out keeps the preset's node, and ``null`` (``None`` in Python) removes a
variable, which is needed when an overridden ``reg_fml`` drops a term. A
dropped term's ``rep_conditions.func`` entry is removed with it; any other
``func`` key that is not a variable on the right-hand side of the formula (a
misspelling, say) is an error rather than silently ignored. The preset's formula, scatter plot and reporting conditions are kept unless you
override them too.

For example, to use the GHI sensors in place of the POA sensors on a stowed
tracker while keeping ``power``, ``t_amb`` and ``w_vel`` as ``e2848_default``
defines them:

.. code-block:: yaml

    captest:
      test_setup: e2848_default
      overrides:
        reg_cols_meas:
          poa: {group: irr_ghi, agg: mean}

or in Python:

.. code-block:: Python

    tst = CapTest.from_params(
        test_setup="e2848_default",
        reg_cols_meas={"poa": {"group": "irr_ghi", "agg": "mean"}},
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
    )

:py:meth:`~captest.CapTest.to_yaml` writes only the terms that differ from the
preset, so a saved file says exactly what was changed. The other overrides are
``reg_fml``, ``rep_conditions`` (merged onto the preset's), ``scatter_plots``
(a name in :data:`~captest.captest.SCATTER_REGISTRY`) and ``params``; the last
three replace the preset's value wholesale.

**Route 3 — assign directly.** Attributes can be set on the instance before
calling :py:meth:`~captest.CapTest.setup`:

.. code-block:: Python

    tst = CapTest(test_setup="custom", ac_nameplate=6_000_000)
    tst.meas = meas
    tst.sim = sim
    tst.reg_cols_meas = my_meas_cols
    tst.reg_cols_sim = my_sim_cols
    tst.reg_fml = "power ~ poa"
    tst.setup()

Checking a setup against your data
----------------------------------
Before :py:meth:`~captest.CapTest.setup` writes any column, it checks that the
resolved setup fits the loaded data: every ``group`` node names a column group,
every ``column`` node names a column, every scalar a calculation requires has a
value, and no calculated column would overwrite a sensor column. All problems are
reported together in one :py:class:`~captest.setup.SetupFitError`, each with its
path in the document. To see the same list without running ``setup()``, call
:py:meth:`~captest.CapTest.check_fit`, which returns an empty list when the setup
fits:

.. code-block:: Python

    tst = CapTest.from_params(
        test_setup="e2848_default",
        reg_cols_meas={"poa": {"group": "irr_ghi", "agg": "mean"}},
        meas=meas,
        sim=sim,
        run_setup=False,
    )
    for error in tst.check_fit():
        print(error.path, error.message)

After ``setup()``, :py:attr:`~captest.CapTest.resolved_setup` holds the complete
:py:class:`~captest.setup.TestSetup` the test used: the preset with your
overrides applied. ``tst.resolved_setup.to_dict()`` is the full document and
``tst.resolved_setup.content_digest()`` its identity, which is equal for two tests
that use the same setup.

In Python, :py:func:`TestSetup.derive <captest.setup.derive>` makes the same
kind of variant from any setup, merging ``reg_cols_meas`` / ``reg_cols_sim`` key
by key and replacing the other fields you pass:

.. code-block:: Python

    from captest import TEST_SETUPS
    from captest.setup import TestSetup

    ghi_variant = TestSetup.derive(
        TEST_SETUPS["e2848_default"],
        name="e2848_ghi",
        reg_cols_meas={"poa": {"group": "irr_ghi", "agg": "mean"}},
    )
