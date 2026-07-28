.. _data_prep:

Preparing Data
==============
Measured and modeled data are sometimes not in the units and dtypes a capacity test
needs. Or, a sensor may be oriented incorrectly and its measurements should be dropped
before aggregation. The **prep stage** is where these types of adjustments can be made.

Prep steps are declared, serialized, and replayed, exactly like filters. The
difference is what they do: a filter selects rows and leaves the values alone,
while a prep step rewrites values (or column identity) in
:py:attr:`~captest.capdata.CapData.data` in place. Each step that runs is
appended to the :py:attr:`~captest.capdata.CapData.prep` list, is written to
the test config by :py:meth:`~captest.captest.CapTest.to_yaml`, and is
re-applied automatically the next time the data is loaded.

Where prep sits in the lifecycle
--------------------------------
The lifecycle of a test is:

**load → prep → setup → filter**

Prep runs after the data is loaded and before
:py:meth:`~captest.captest.CapTest.setup`. That ordering is deliberate:
``setup`` resolves regression columns and calculates derived parameters (cell
temperature, spectral corrections) from the values in ``data``, so the values
must already be in the units those calculations assume. Filtering comes last,
because every filter threshold is evaluated against prepped values.

The prep stage is the only place a data adjustment is *recorded*. Editing
``cd.data`` directly in a notebook still works and always has, but nothing
captures that it happened: the exported yaml does not show it, and
:py:meth:`~captest.captest.CapTest.reload` re-invokes the loader and silently
discards it. A reviewer reading an exported config can see a recorded prep
step; they cannot see a hand-edit.

Prep is not reported by :py:meth:`~captest.capdata.CapData.get_summary` or
:py:meth:`~captest.capdata.CapData.describe_filters`. Those report point
counts, and prep removes no points. Use
:py:meth:`~captest.capdata.CapData.describe_prep` interactively and the
``meas_prep`` / ``sim_prep`` block of the config as the durable record.

The prep methods
----------------
:py:class:`~captest.capdata.CapData` provides four prep wrappers, each of
which builds a step from :py:mod:`captest.prep`, runs it against ``data``, and
appends it to :py:attr:`~captest.capdata.CapData.prep`:

- :py:meth:`~captest.capdata.CapData.prep_convert_units` — convert between
  units from a supported table.
- :py:meth:`~captest.capdata.CapData.prep_scale` — apply
  ``value * factor + offset``.
- :py:meth:`~captest.capdata.CapData.prep_astype` — cast columns to a dtype.
- :py:meth:`~captest.capdata.CapData.prep_custom` — run an arbitrary callable.

A typical measured side needs the temperatures and the wind speed converted,
and a PVsyst side needs its power column rescaled:

.. code-block:: Python

    import captest as ct

    das = ct.load_data('./data/')
    das.prep_convert_units(group_regex='^temp', from_units='F', to_units='C')
    das.prep_convert_units(group='wind', from_units='mph', to_units='m/s')

    sim = ct.load_pvsyst('./pvsyst/VC1_HourlyRes_1.CSV')
    sim.prep_scale(columns=['E_Grid'], factor=0.001)

Within a :py:class:`~captest.captest.CapTest`, load the data without running
``setup``, prep each side, then call
:py:meth:`~captest.captest.CapTest.setup`:

.. code-block:: Python

    tst = ct.CapTest.from_params(
        test_setup='e2848_default',
        meas_path='./data/',
        sim_path='./pvsyst/VC1_HourlyRes_1.CSV',
        run_setup=False,
    )
    tst.meas.prep_convert_units(group_regex='^temp', from_units='F', to_units='C')
    tst.meas.prep_convert_units(group='wind', from_units='mph', to_units='m/s')
    tst.sim.prep_scale(columns=['E_Grid'], factor=0.001)
    tst.setup()

:py:meth:`~captest.capdata.CapData.describe_prep` returns a written summary of
what was applied:

.. code-block:: Python

    print(tst.meas.describe_prep())

.. code-block:: text

    Columns met1_mod_temp1, met1_mod_temp2, met2_mod_temp1, met2_mod_temp2, met1_amb_temp, met2_amb_temp were converted from F to C.
    Columns met1_windspeed, met2_windspeed were converted from mph to m/s.

Every step is atomic. If a step raises, ``data`` and ``column_groups`` are
restored to their pre-step state and nothing is appended to
:py:attr:`~captest.capdata.CapData.prep`, so a failed step is a no-op rather
than a half-applied change.

.. _prep-selectors:

Selecting columns
-----------------
``prep_convert_units``, ``prep_scale``, and ``prep_astype`` take exactly one
of three column selectors:

``columns``
    An explicit list of column names, acted on in the order given.

``group``
    A :py:attr:`~captest.capdata.CapData.column_groups` id, or a list of ids.

``group_regex``
    A regular expression matched against the ``column_groups`` ids. For
    example ``'^temp'`` reaches ``temp_amb``, ``temp_bom``, and ``temp_mod``
    in a single step.

The group selectors expand to column names in ``column_groups`` order, with
duplicates removed. Passing more than one selector, or none, is an error:

.. code-block:: Python

    das.prep_scale(factor=0.001)
    # ValueError: Scale requires exactly one of 'columns', 'group', or
    # 'group_regex'; got none.

Selectors are resolved **at run time**, not when the step is constructed. A
step that runs after an earlier drop or rename sees the current columns, which
is what makes a recorded chain replay correctly: a rename recorded before a
conversion is replayed first, and the conversion resolves against the new
names.

A selector that resolves to zero columns is always an error rather than a
silent no-op, and the messages name what is available:

.. code-block:: Python

    das.prep_convert_units(group='temp_ambient', from_units='F', to_units='C')
    # ValueError: Column group 'temp_ambient' is not in column_groups.
    # Available ids: [...]. Did you mean 'temp_amb'?

The remaining steps select differently:
:py:meth:`~captest.capdata.CapData.drop_cols` takes its column names directly,
:py:meth:`~captest.capdata.CapData.rename_cols` takes them from the keys of
its ``column_map``, and :py:meth:`~captest.capdata.CapData.prep_custom` takes
no selector at all — the callable decides what it touches (see
`Custom prep steps`_).

Converting units
----------------
:py:meth:`~captest.capdata.CapData.prep_convert_units` applies an affine
conversion, ``value * factor + offset``, to the selected columns and writes
the result back to the same column names. The supported pairs are:

.. list-table::
   :header-rows: 1

   * - From
     - To
   * - ``F``
     - ``C``
   * - ``K``
     - ``C``
   * - ``mph``
     - ``m/s``
   * - ``km/h``
     - ``m/s``
   * - ``kn``
     - ``m/s``
   * - ``ft/s``
     - ``m/s``
   * - ``W``
     - ``kW``
   * - ``MW``
     - ``kW``
   * - ``mbar``
     - ``Pa``
   * - ``m``
     - ``cm``
   * - ``in``
     - ``cm``

Only the forward direction is tabulated; the **inverse of every pair is
derived automatically**, so ``from_units='C', to_units='F'`` works without a
second table entry. Unit names are normalized case-insensitively through a
table of aliases, so ``'degF'``, ``'fahrenheit'``, and ``'F'`` are the same
unit, as are ``'kph'`` and ``'km/h'``, or ``'knots'`` and ``'kn'``.

Requesting a pair that is not supported in either direction raises and lists
what is:

.. code-block:: Python

    das.prep_convert_units(columns=['irr_poa_1'], from_units='W/m2', to_units='kW/m2')
    # ValueError: No conversion from 'W/m2' to 'kW/m2'. Inverses of the
    # supported pairs are derived automatically; supported pairs: [...].
    # Use Scale(factor=..., offset=...) for anything else.

Anything the table does not cover is expressed directly with
:py:meth:`~captest.capdata.CapData.prep_scale`, which is the raw-numbers
escape hatch:

.. code-block:: Python

    sim.prep_scale(columns=['E_Grid'], factor=0.001)
    das.prep_scale(group='irr_poa', factor=1000)

.. note::

    A unit conversion is an *asserted* transformation, not a checked one.
    Nothing on :py:class:`~captest.capdata.CapData` tracks the current units
    of a column, and no unit is inferred from a column name. Converting a
    column that is already in the target units will silently produce wrong
    values, which is why an identical repeat of a step is refused (see
    :ref:`prep-once-per-load`).

Both :py:meth:`~captest.capdata.CapData.prep_convert_units` and
:py:meth:`~captest.capdata.CapData.prep_scale` require numeric columns and
raise a ``TypeError`` naming the first offender otherwise. A frame that loaded
as ``object`` — a PVsyst export with a stray index column is the usual cause —
is fixed with :py:meth:`~captest.capdata.CapData.prep_astype` first:

.. code-block:: Python

    sim.drop_cols(['index'])
    sim.prep_astype(columns=['E_Grid'], dtype='float64')

Dropping and renaming columns
-----------------------------
:py:meth:`~captest.capdata.CapData.drop_cols` and
:py:meth:`~captest.capdata.CapData.rename_cols` keep the names and signatures
they have always had, and now record a prep step by default. There are no
separate ``prep_drop_columns`` / ``prep_rename_columns`` methods:

.. code-block:: Python

    sim.drop_cols(['index'])
    das.rename_cols({'met1_poa_pyran': 'irr_poa_met1'})

Both accept ``record=False``, which performs the change without recording it.
That option exists for **internal callers** inside the library — for example
:py:meth:`~captest.capdata.CapData.agg_sensors`, which renames the aggregate
columns it creates and must not put that rename into the user's prep chain,
because replaying it on the next load would name columns that do not yet
exist. As a user, leave ``record`` alone.

.. _prep-once-per-load:

Prep runs once per load
-----------------------
Prep steps mutate ``data``, so they are **not idempotent**: converting °F to
°C twice is silently wrong rather than loudly broken. Filters do not have this
problem because replaying a pipeline resets the chain first; prep has no
equivalent reset, because ``data`` is the only copy.

Three consequences follow:

- ``prep_convert_units``, ``prep_scale``, and ``prep_astype`` raise a
  ``RuntimeError`` when a step equal to one already applied — same type, same
  arguments, ``custom_name`` aside — is run again. Two *different* conversions
  of the same columns are allowed; it is the exact repeat that is refused.
  The check compares the selector as written, so the same column reached two
  ways (``columns=["poa_1"]`` and ``group="poa"``) is not caught.
- :py:meth:`~captest.capdata.CapData.run_prep` refuses to replay a config onto
  a :py:class:`~captest.capdata.CapData` whose prep chain is already
  populated.
- There is no ``reset_prep()``. Nothing can undo a mutation, so re-prepping
  means going back to raw data with
  :py:meth:`~captest.captest.CapTest.reload`:

  .. code-block:: Python

      tst.reload('sim')

  ``reload`` snapshots the outgoing side's applied prep chain into
  ``sim_prep``, loads a fresh un-prepped frame from the stored path, replays
  that prep, and then runs ``setup`` for that side. It re-applies **the same
  steps** — including after ``tst.reload('sim', path=...)`` points the side at
  a different file, which is the case it exists for. Because the snapshot
  overwrites ``sim_prep`` with the applied chain, editing ``tst.sim_prep``
  and then reloading does not change anything. To run a *different* prep,
  rebuild the test from its paths — with
  :py:meth:`~captest.captest.CapTest.from_yaml` on an edited config, or
  ``from_params(..., run_setup=False)`` followed by the ``prep_*`` calls you
  want.

:py:meth:`~captest.capdata.CapData.run_prep` is transactional across the whole
batch, not just per step: if any step raises, ``data``, ``column_groups``, and
the prep chain are restored to their state before the call and a note naming
the failing step is attached to the exception. A partially prepped frame is
never left behind.

.. note::

    When a :py:class:`~captest.captest.CapTest` is built from a *pre-built*
    :py:class:`~captest.capdata.CapData` rather than a path, a stored
    ``meas_prep`` / ``sim_prep`` config is kept but **not** applied — the
    object may already have been prepped, and prep is not idempotent. A
    ``UserWarning`` reports how many steps were skipped and names the call
    that applies them: ``tst.meas.run_prep(tst.meas_prep)``.

Prep comes before filtering
---------------------------
A filter that has already run was evaluated against un-prepped values, so
rewriting those values afterwards produces wrong numbers rather than a crash.
Prep steps that rewrite values refuse to run once filters are applied:

.. code-block:: Python

    das.filter_irr(200, 950)
    das.prep_convert_units(group='wind', from_units='mph', to_units='m/s')
    # RuntimeError: Cannot run ConvertUnits after filtering — 1 filters are
    # already applied and were evaluated against un-prepped values. Call
    # cd.reset_filter() first, or reload this side.

The recovery is :py:meth:`~captest.capdata.CapData.reset_filter`, which clears
the filter chain and leaves the prep chain intact:

.. code-block:: Python

    das.reset_filter()
    das.prep_convert_units(group='wind', from_units='mph', to_units='m/s')

Column drops and renames are exempt. They change column identity rather than
values, so :py:meth:`~captest.capdata.CapData.drop_cols` and
:py:meth:`~captest.capdata.CapData.rename_cols` stay legal after filtering.

Prep before aggregating sensors
-------------------------------
:py:meth:`~captest.capdata.CapData.agg_sensors` combines a group of sensors
into one representative column — typically the mean of several thermocouples
or pyranometers. Averaging across columns that are in *different* units
produces a meaningless number, and no error is raised, because nothing tracks
the units of a column.

This matters in practice because ``agg_sensors`` is often called by hand after
``setup``, while prep runs before it. Convert units in the prep stage, before
aggregating:

.. code-block:: Python

    tst.meas.prep_convert_units(group='temp_bom', from_units='F', to_units='C')
    tst.setup()
    tst.meas.agg_sensors()

If one met station reports °F and another °C, convert the °F columns
explicitly by name — a per-column ``columns=[...]`` selector — before the two
groups are aggregated together.

Prep in the config file
-----------------------
:py:meth:`~captest.captest.CapTest.to_yaml` writes the applied prep chain for
each side under the ``meas_prep`` and ``sim_prep`` keys, alongside the
``meas_filters`` / ``sim_filters`` pipelines described in
:ref:`reproducing-a-test`. The keys are omitted entirely when a side has no
prep.

.. code-block:: yaml

    captest:
      test_setup: e2848_default
      meas_path: ./data/
      sim_path: ./pvsyst/VC1_HourlyRes_1.CSV
      # ... test settings ...
      meas_prep:
      - type: ConvertUnits
        columns: null
        custom_name: null
        from_units: F
        group: null
        group_regex: ^temp
        to_units: C
      - type: ConvertUnits
        columns: null
        custom_name: null
        from_units: mph
        group: wind
        group_regex: null
        to_units: m/s
      sim_prep:
      - type: DropColumns
        columns:
        - index
        custom_name: null
        group: null
        group_regex: null
      - type: Scale
        columns:
        - E_Grid
        custom_name: null
        factor: 0.001
        group: null
        group_regex: null
        offset: 0.0
      meas_filters:
      # ...

Unlike the filter pipelines, which are stored as *pending* and replayed by
:py:meth:`~captest.captest.CapTest.run_test`, prep is applied **at load**.
:py:meth:`~captest.captest.CapTest.from_yaml` loads each side from its path
and immediately replays that side's prep, before ``setup``:

.. code-block:: Python

    tst = ct.CapTest.from_yaml('./project.yaml')
    print(tst.sim.describe_prep())

.. code-block:: text

    Columns index were dropped.
    Columns E_Grid were scaled by 0.001 with an offset of 0.0.

The same replay happens for :py:meth:`~captest.captest.CapTest.from_params`
and :py:meth:`~captest.captest.CapTest.from_mapping` whenever a side is built
from a path, including under ``run_setup=False``, and on every
:py:meth:`~captest.captest.CapTest.reload`. That is what makes a reload with a
new file safe: the new data gets the same adjustments the old data had.

A single :py:class:`~captest.capdata.CapData` can be round-tripped on its own
with :py:meth:`~captest.capdata.CapData.prep_to_config` and
:py:meth:`~captest.capdata.CapData.run_prep`, which mirror
:py:meth:`~captest.capdata.CapData.filters_to_config` and
:py:meth:`~captest.capdata.CapData.run_pipeline`:

.. code-block:: Python

    config = das.prep_to_config()
    fresh = ct.load_data('./data/')
    fresh.run_prep(config)

Custom prep steps
-----------------
:py:meth:`~captest.capdata.CapData.prep_custom` is the escape hatch for
adjustments the declarative vocabulary does not cover. The callable
receives the :py:class:`~captest.capdata.CapData` as its first argument and
mutates ``data`` in place; its return value is ignored.

.. code-block:: Python

    # in project_prep.py
    def blank_before_sunrise(cd, columns, hour):
        mask = cd.data.index.hour < hour
        cd.data.loc[mask, columns] = float('nan')

.. code-block:: Python

    from project_prep import blank_before_sunrise

    das.prep_custom(blank_before_sunrise, ['met1_poa_east'], 7)

``func`` serializes as a module-qualified name, so it **must be a
module-level function** that can be imported when the config is replayed — the
same constraint that applies to a custom filter. A lambda or a locally defined
closure raises when the config is written, not when the step is run.

``custom_name`` is keyword-only on
:py:meth:`~captest.capdata.CapData.prep_custom` so it cannot collide with an
argument destined for ``func``. ``prep_convert_units``, ``prep_scale``, and
``prep_astype`` accept it too, as a display label in
:py:meth:`~captest.capdata.CapData.describe_prep`;
:py:meth:`~captest.capdata.CapData.drop_cols` and
:py:meth:`~captest.capdata.CapData.rename_cols` do not.

A custom step takes no column selector. Every argument other than
``custom_name`` is forwarded to ``func``, so
``prep_custom(fn, columns=['a'])`` passes ``columns=['a']`` to ``fn`` rather
than selecting columns, and the serialized step records it as one of ``func``'s
keyword arguments. ``func`` decides for itself which columns it touches, and
whatever validation of them matters is ``func``'s to do.
