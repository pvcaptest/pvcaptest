.. _reporting_conditions:

====================
Reporting Conditions
====================
An ASTM E2848 capacity test compares the measured capacity to the modeled
capacity at a single, agreed-upon set of **reporting conditions** (RCs) — the
representative irradiance, temperature, and wind speed the plant is rated at.
Because both regressions are evaluated at the *same* point, pvcaptest models the
reporting conditions as one value owned by the test:
:py:attr:`~captest.captest.CapTest.rc`.

This page explains how the single test RC is established, overridden, tracked,
used in filtering and results, and persisted across a yaml round-trip. It
assumes a :py:class:`~captest.captest.CapTest` instance ``ct`` has been created
and set up as described in :ref:`captest`.

The single test reporting conditions
------------------------------------
:py:attr:`~captest.captest.CapTest.rc` is a one-row
:class:`pandas.DataFrame` (or ``None`` before any have been established) holding
one value per regression variable:

.. code-block:: Python

    >>> ct.rc
         poa  t_amb  w_vel
    0  805.1   24.7    2.1

Its provenance is tracked by :py:attr:`~captest.captest.CapTest.rc_source`,
which is one of ``'meas'``, ``'sim'``, or ``'manual'`` — recording whether the
conditions were computed from the measured data, computed from the modeled data,
or supplied directly.

.. note::

    A standalone :py:class:`~captest.capdata.CapData` used without a
    ``CapTest`` keeps its own ``cd.rc`` and is unaffected by the test-level
    ownership described here. Inside a ``CapTest`` the test RC is authoritative
    and ``ct.meas`` / ``ct.sim`` resolve reporting values from it.

Computing reporting conditions
------------------------------
:py:meth:`~captest.captest.CapTest.rep_cond` computes the reporting conditions
from one dataset using the selected test setup's default aggregations. For the
standard E2848 setup, POA uses the 60th percentile of the filtered POA while
ambient temperature and wind speed use the mean.

.. code-block:: Python

    ct.rep_cond()              # compute from measured data (rc_source -> 'meas')
    ct.rep_cond(which='sim')   # compute from modeled data  (rc_source -> 'sim')

Whichever side it is computed on becomes the single test RC: the call updates
:py:attr:`~captest.captest.CapTest.rc` and sets
:py:attr:`~captest.captest.CapTest.rc_source` accordingly. Calling
``ct.meas.rep_cond()`` (or ``ct.sim.rep_cond()``) directly has the same effect —
any reporting-conditions calculation on a test member flows up to ``ct.rc``
(last writer wins). With no ``which`` argument, ``ct.rep_cond()`` defaults to the
current ``rc_source``, so re-running it does not silently switch sides.

When a computation *changes* the source — for example computing from ``sim``
after the RC was previously taken from ``meas``, or overwriting a computed RC
with a manual one — a ``UserWarning`` is emitted so an unintended switch is
visible. Re-computing on the same side with an unchanged result is silent.

A write that *changes the RC values* also checks for stale RC-dependent
filters: any applied ``filter_irr`` step with ``ref_val='rep_irr'`` (or
``'self_val'``) resolved its irradiance window against the *previous*
reporting conditions and no longer matches. When such steps exist, the same
single ``UserWarning`` names them (e.g. ``meas.filters[5] (Irradiance)``) so
they can be re-run — for example with
:py:meth:`~captest.capdata.CapData.rerun_from` — against the new conditions.
:py:meth:`~captest.captest.CapTest.run_test` re-runs both pipelines itself, so
it suppresses this notice for the steps it is about to replay.

Anchoring irradiance filters on the reporting irradiance
--------------------------------------------------------
After the reporting conditions are computed it is common to apply a second,
narrower irradiance filter around the reporting irradiance.
:py:attr:`~captest.captest.CapTest.rep_irr_filter_low` and
:py:attr:`~captest.captest.CapTest.rep_irr_filter_high` provide the fractional
bounds (``0.8`` and ``1.2`` with the default ``rep_irr_filter=0.2``), so the
same window is applied consistently to the measured and modeled data:

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

Passing ``ref_val='rep_irr'`` resolves the reference irradiance from the single
test RC: within a ``CapTest`` both ``ct.meas`` and ``ct.sim`` read
:py:attr:`~captest.capdata.CapData.rep_irr` from ``ct.rc``. This means a modeled
filter can anchor on the test's reporting irradiance even when the conditions
were computed from the measured data — without passing the value by hand. If no
test RC has been established, ``ref_val='rep_irr'`` raises a ``ValueError``
directing you to compute or set the reporting conditions first.

Setting reporting conditions manually
-------------------------------------
Sometimes the reporting conditions should be supplied directly rather than
computed — for a sensitivity study, or to reproduce a reviewing party's stated
values. Assigning to :py:attr:`~captest.captest.CapTest.rc` is the single public
way to do this; it records ``rc_source='manual'``:

.. code-block:: Python

    # a one-row DataFrame, a Series, or a dict of regression variable -> value
    ct.rc = {'poa': 800.0, 't_amb': 25.0, 'w_vel': 2.0}

The value must provide a number for every right-hand-side variable of the
(shared measured/modeled) regression formula; interaction terms such as
``I(poa * t_amb)`` are unwrapped to their component variables. Extra columns are
preserved. The assignment validates the input and raises:

- ``RuntimeError`` if :py:meth:`~captest.captest.CapTest.setup` has not run (the
  regression formula is unknown);
- ``ValueError`` if the measured and modeled formulas differ, if the value
  resolves to more than one row, or if a required regression variable is
  missing (the message names the missing variables);
- ``TypeError`` if the value is not a DataFrame, Series, or dict.

A manual RC is treated as the authoritative value: it is not overwritten by
pipeline replay — on load or during :py:meth:`~captest.captest.CapTest.run_test`
(replayed RepCond steps compute side-local RCs only) — and a later ``rep_cond``
call that would change the source back to a computed value emits the
source-change warning.

Reporting conditions and results
--------------------------------
:py:meth:`~captest.captest.CapTest.captest_results` (and
:py:meth:`~captest.captest.CapTest.captest_results_check_pvalues`) predict both
the measured and modeled regressions at the single test RC ``ct.rc``. The
returned :py:class:`~captest.captest.CapTestResults` object carries the
reporting conditions the predictions were made at (``results.rc``) and their
provenance (``results.rc_source``) alongside the capacity ratio. If no
reporting conditions have been established, ``captest_results`` raises a
``ValueError`` — call ``ct.rep_cond(...)`` or assign ``ct.rc`` first.

Saving and restoring reporting conditions
-----------------------------------------
The single test RC round-trips through
:py:meth:`~captest.captest.CapTest.to_yaml` /
:py:meth:`~captest.captest.CapTest.from_yaml` (see :ref:`saving_reproducing`):

- **Computed** reporting conditions are not value-serialized. They are recomputed
  on load by replaying the ``RepCond`` step in the ``rc_source`` side's filter
  pipeline, so the restored RC reflects the same filtered data.
- **Manual** reporting conditions carry their values in a
  ``reporting_conditions_values`` block. On load they are re-validated and seeded
  before the filter pipelines replay, so a self-anchoring ``ref_val='rep_irr'``
  filter resolves correctly.

On load the configured ``rc_source`` side's pipeline is replayed first, so a
cross-side ``ref_val='rep_irr'`` filter (for example a measured filter anchored on
a modeled reporting irradiance) resolves against an already-established ``ct.rc``.

.. note::

    Only the ``rc_source`` side's pipeline should contain a reporting-conditions
    step. A configuration with a ``RepCond`` step in *both* pipelines under a
    computed ``rc_source`` is ambiguous — whichever side replays second would
    overwrite the test reporting conditions and flip ``rc_source`` — so loading
    or re-running such a configuration emits a ``UserWarning`` asking for the
    ``RepCond`` step to be removed from the non-``rc_source`` pipeline.

Adjusting the default aggregation
---------------------------------
The reporting-condition *recipe* — how each regression variable is aggregated —
comes from the selected test setup and can be customized per call (e.g.
``ct.rep_cond(func={'poa': perc_wrap(55)})``) or in the config file. See
:ref:`captest` for the ``rep_conditions`` configuration and percentile helpers.

See also
--------
- :py:attr:`~captest.captest.CapTest.rc`,
  :py:attr:`~captest.captest.CapTest.rc_source`, and
  :py:meth:`~captest.captest.CapTest.rep_cond` in the
  :doc:`CapTest API reference <../source/api_reference/captest>`.
- :py:meth:`~captest.capdata.CapData.rep_cond` and
  :py:attr:`~captest.capdata.CapData.rep_irr` in the
  :doc:`CapData API reference <../source/api_reference/capdata>`.
- :ref:`saving_reproducing` for the full configuration round-trip.
