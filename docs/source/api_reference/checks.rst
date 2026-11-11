.. currentmodule:: captest

Validity Checks
===============

The ``captest.checks`` module holds validity checks: requirements a dataset
must meet, such as a minimum number of high-irradiance intervals or days. A
check reports a :py:class:`~captest.checks.CheckResult`; it never removes rows.
The ``CapData.check_*`` methods are thin wrappers that build the matching check
and register it on ``CapData.checks`` without evaluating it.
:py:meth:`~captest.capdata.CapData.evaluate_checks` evaluates everything
registered. ``CapTest.to_yaml`` and ``from_yaml`` do not yet carry the checks registered on
``CapTest.meas`` and ``CapTest.sim``; that arrives with CapTest PR-test support.
Until then, save and restore them with
:py:meth:`~captest.capdata.CapData.checks_to_config` and
:py:meth:`~captest.capdata.CapData.set_checks`. Checks may also be built and
evaluated directly:

.. code-block:: Python

   from captest import checks

   check = checks.IntervalCount(400, 60)
   result = check.evaluate(cd, checks.CheckContext(dt_hours=0.25))
   result.passed, result.summary

.. warning::

   Loading a config that contains a custom check imports and calls the
   callable it names. Only load configs from trusted sources.

Context and Results
-------------------

.. autosummary::
   :toctree: generated/

   checks.CheckContext
   checks.CheckResult

Checks
------

.. autosummary::
   :toctree: generated/

   checks.BaseCheck
   checks.IntervalCount
   checks.Duration
   checks.ContinuousHoursDays
   checks.DailyInsolationDays
   checks.NrelMinimumIrradiance
   checks.AnyOf
   checks.CustomCheck

Registry and Serialization
--------------------------

.. data:: captest.checks.CHECK_REGISTRY

   Check class names mapped to their classes, used to rebuild a check from its
   config.

.. autosummary::
   :toctree: generated/

   checks.check_from_config
