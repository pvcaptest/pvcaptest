.. _custom_test_setups:

Custom Test Setups
==================
The built-in ``test_setup`` presets (``e2848_default``, ``bifi_e2848_etotal``,
``bifi_power_tc_meas_tbom``, ``bifi_power_tc_calc_tbom``, ``e2848_spec_corrected_poa``)
cover the most common capacity-test configurations. When a project calls for a
different regression equation or a non-standard column calculation, pvcaptest lets
you supply your own ``reg_cols_meas`` and ``reg_cols_sim`` dicts and user-defined
parameter calculation functions without modifying the package.

This page explains the structure of those dicts, shows how the built-in
:py:mod:`captest.calcparams` functions plug into them, and describes the three
ways to wire a custom dict into a :py:class:`~captest.captest.CapTest`.

The regression column dictionary grammar
------------------------------------
Each key in ``reg_cols_meas`` or ``reg_cols_sim`` maps a regression term (such
as ``"power"`` or ``"poa"``) to one of three node forms.

The simplest node is a plain string matching a column name in the ``data`` attribute.
For example, this is the approach used in the built-in test setups for the 
PVsyst output:

.. code-block:: Python

    "poa": "GlobInc"

A **simple aggregation** is a two-element tuple of the column-group id and
an aggregation function name. This will be aggregated using the ``CapData.agg_group``
method. This example assumes that you have a key in your ``CapData.column_groups`` called
``irr_poa`` that points to a list of columns, which contain measurements from POA
irradiance sensors. See the :ref:`Column Grouping <col-grouping>` section of the CapData
workflow documentation page for additional explanation.

.. code-block:: Python

    "poa": ("irr_poa", "mean")

.. note::

    Aggregation uses `pandas.DataFrame.agg <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html>`_
    which accepts a wide range of aggregation methods.

A **calculated column** is a two-element tuple of a callable and a dict
mapping the callable's keyword arguments to column names, aggregation tuples,
or nested calculated-column tuples:

.. code-block:: Python

    "poa": (
        e_total,
        {
            "poa": ("irr_poa", "mean"),
            "rpoa": ("irr_rpoa", "mean"),
        },
    )

Nesting is allowed to any depth. During
:py:meth:`~captest.captest.CapTest.setup`, pvcaptest recursively walks the tree bottom-up:
each ``(func, kwargs_dict)`` tuple creates a new column named
``func.__name__`` in ``CapData.data``, and that name is passed upward as input
to any parent tuple. For the example above, ``e_total`` is a callable (a function from
the ``calcprams`` module) and processing this portion of the dictionary adds a column
to the ``data`` attribute with the name ``e_total``. Also, the dictionary is updated
so that ``poa`` now points to ``e_total``.

Using calcparams functions
--------------------------
The functions in :py:mod:`captest.calcparams` are the public building blocks
for calculated columns. Import the ones you need:

.. code-block:: Python

    from captest.calcparams import (
        e_total,
        bom_temp,
        cell_temp,
        power_temp_correct,
        rpoa_pvsyst,
        scale,
    )

Using custom functions
----------------------
If you need to calculate a parameter which does not have a function in the ``calcparms``
module, you can write your own.

The dictionary accepts plain Python functions that are not part of
:py:mod:`captest.calcparams`, as long as they follow the same signature
convention:

- First positional argument must be ``data``, the source DataFrame.
- Remaining arguments are column names passed as strings.
- The function returns a :class:`pandas.Series` indexed like ``data``.
- The function must be defined with ``def`` (not a ``lambda``) so it has a
  ``__name__``.

.. code-block:: Python

    def my_adjusted_poa(data, poa=None, adjustment=1.0, verbose=True):
        """Scale a POA column by a site-specific adjustment factor."""
        if verbose:
            print(f"Calculating my_adjusted_poa as {poa} * {adjustment}")
        return data[poa] * adjustment

    my_meas_cols = {
        "poa": (
            my_adjusted_poa,
            {"poa": ("irr_poa", "mean"), "adjustment": 1.05},
        ),
        ...
    }

Adding a verbose kwarg to print an explanation of the calculation is not required, but
strongly recommended as it makes the calculation traceable for a reviewing party.

.. note::

    Each function name must be unique within a single ``reg_cols_meas`` or
    ``reg_cols_sim`` dict because the column added to ``CapData.data`` is
    always named ``func.__name__``. If two nodes call functions with the same
    name, the second call overwrites the column produced by the first.

Creating a Custom Regression Columns Dictionary
-----------------------------------------------
The example below builds measured and modeled column dicts that compute
temperature-corrected power from raw power, back-of-module temperature
(estimated from POA, ambient temperature, and wind speed), and cell
temperature:

.. code-block:: Python

    from captest.calcparams import bom_temp, cell_temp, power_temp_correct

    my_meas_cols = {
        "power": (
            power_temp_correct,
            {
                "power": ("real_pwr_mtr", "sum"),
                "cell_temp": (
                    cell_temp,
                    {
                        "poa": ("irr_poa", "mean"),
                        "bom": (
                            bom_temp,
                            {
                                "poa": ("irr_poa", "mean"),
                                "temp_amb": ("temp_amb", "mean"),
                                "wind_speed": ("wind_speed", "mean"),
                            },
                        ),
                    },
                ),
            },
        ),
        "poa": ("irr_poa", "mean"),
    }

    my_sim_cols = {
        "power": (
            power_temp_correct,
            {"power": "E_Grid", "cell_temp": "TArray"},
        ),
        "poa": "GlobInc",
    }

Scalar auto-injection
---------------------
Scalar parameters such as ``power_temp_coeff``, ``base_temp``,
``bifaciality``, and ``spectral_module_type`` do not need to appear in the
dict. When a :py:mod:`captest.calcparams` function has a keyword argument
whose name matches an attribute on the ``CapData`` instance, pvcaptest injects
that value automatically. :py:class:`~captest.captest.CapTest` propagates
these scalars onto both ``CapData`` instances during
:py:meth:`~captest.captest.CapTest.setup`, so setting them on the
:py:class:`~captest.captest.CapTest` instance is sufficient:

.. code-block:: Python

    ct = CapTest.from_params(
        test_setup="custom",
        reg_cols_meas=my_meas_cols,
        reg_cols_sim=my_sim_cols,
        reg_fml="power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        power_temp_coeff=-0.36,   # injected automatically into power_temp_correct
        base_temp=25,             # injected automatically into power_temp_correct
    )

To override the auto-injected value for a specific node, include the scalar
explicitly in that node's kwarg dict.

This approach is recommended because it ensures values tha should be consistent between, 
the measured data and simulated data, like ``bifacility``, match.

Wiring a custom dict into CapTest
----------------------------------
There are three equivalent ways to supply custom column dicts.

**Route 1 — fully custom setup.** Pass ``test_setup='custom'`` with all three
required overrides. ``scatter_plots`` and ``rep_conditions`` default to
``scatter_default`` and an empty dict if omitted:

.. code-block:: Python

    from captest import CapTest

    ct = CapTest.from_params(
        test_setup="custom",
        reg_cols_meas=my_meas_cols,
        reg_cols_sim=my_sim_cols,
        reg_fml="power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance="- 4",
    )

**Route 2 — override a named preset.** If a built-in preset's formula is
correct but the column mappings need to change, pass ``reg_cols_meas`` and / or
``reg_cols_sim`` alongside a named ``test_setup``. The preset's formula,
scatter plot, and reporting conditions are inherited:

.. code-block:: Python

    ct = CapTest.from_params(
        test_setup="e2848_default",
        reg_cols_meas=my_meas_cols,
        reg_cols_sim=my_sim_cols,
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
    )

**Route 3 — assign directly.** Attributes can be set on the instance before
calling :py:meth:`~captest.captest.CapTest.setup`:

.. code-block:: Python

    ct = CapTest(test_setup="custom", ac_nameplate=6_000_000)
    ct.meas = meas
    ct.sim = sim
    ct.reg_cols_meas = my_meas_cols
    ct.reg_cols_sim = my_sim_cols
    ct.reg_fml = "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
    ct.setup()

