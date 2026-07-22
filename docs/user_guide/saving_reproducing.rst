.. _saving_reproducing:

============================
Saving and Reproducing Tests
============================
A :py:class:`~captest.captest.CapTest` can write its full configuration — the
test settings and the filter pipelines applied to the measured and modeled
data — to a single yaml file, and reload that file to reproduce the test
exactly. This makes a capacity test portable: it can be archived, shared,
reviewed, and validated from one file.

These sections assume a ``CapTest`` instance ``ct`` has been created and run as
described in :ref:`captest`.

Saving a test setup
-------------------
:py:meth:`~captest.captest.CapTest.to_yaml` writes the main test settings back
to a yaml file. This can be useful after adjusting a setup in a notebook and
wanting to save the settings for a future run.

.. code-block:: Python

    ct.to_yaml('./project.yaml')

By default, ``to_yaml`` updates the selected section of an existing yaml file
and preserves other top-level sections, such as project metadata. It writes the
test settings, data paths, and the filter pipelines applied to the measured and
modeled data (see :ref:`reproducing-a-test` below). It does not write the
measured data, modeled data, fitted regression results, or plots.

:py:meth:`~captest.captest.CapTest.to_mapping` returns the same configuration
as a plain dictionary instead of writing a file. It is the symmetric inverse
of :py:meth:`~captest.captest.CapTest.from_mapping` and is useful when the
configuration should be inspected, stored, or transformed programmatically
rather than written straight to yaml.

.. _reproducing-a-test:

Reproducing a complete test from a config file
----------------------------------------------
A :py:meth:`~captest.captest.CapTest.to_yaml` config file captures more than the
test settings — it also records the **filter pipeline applied to each dataset**.
Every filter step run on ``ct.meas`` and ``ct.sim``, including the reporting-
conditions calculation, is serialized under the ``meas_filters`` and
``sim_filters`` keys. This makes the yaml file a complete, portable record of a
capacity test: a colleague can reproduce, review, or validate the test from the
single file, without access to the original notebook.

Because the filters are recorded automatically as they are applied, the only
extra step is calling ``to_yaml`` once the test has been run:

.. code-block:: Python

    ct.to_yaml('./project.yaml')

The resulting file lists each step with its arguments. For example, the
measured-side pipeline written by the bifacial example notebook:

.. code-block:: yaml

    captest:
      test_setup: bifi_e2848_etotal_rear_shade_sim
      meas_path: ./data/example_meas_data_bifi.csv
      sim_path: ./data/pvsyst_example_HourlyRes_2_bifi.CSV
      # ... test settings ...
      meas_filters:
      - type: Custom
        func: pandas.core.frame:DataFrame.dropna
        args: []
        kwargs: {}
        custom_name: null
      - type: Irradiance
        low: 400
        high: 1400
        ref_val: null
        col_name: null
        custom_name: null
      - type: Outliers
        envelope_kwargs: null
        custom_name: null
      - type: Regression
        n_std: 2
        custom_name: null
      - type: RepCond
        func: null
        percent_filter: 20
        # ... reporting-condition settings ...
      - type: Irradiance
        low: 0.8
        high: 1.2
        ref_val: rep_irr
        col_name: null
        custom_name: null

Loading the file with :py:meth:`~captest.captest.CapTest.from_yaml` loads the
measured and modeled data and runs
:py:meth:`~captest.captest.CapTest.setup` to assign the regression and
calculated columns. The recorded filter pipelines are **not** applied at load
— they are stored on the instance as ``tst.meas_filters_pending`` and
``tst.sim_filters_pending``, so a freshly loaded test holds unfiltered data,
ready for review:

.. code-block:: Python

    tst = CapTest.from_yaml('./project.yaml')

Running the test replays the pending pipelines and reproduces the complete
test — filtering, reporting conditions, both regression fits, and the results
— in a single line:

.. code-block:: Python

    results = CapTest.from_yaml('./project.yaml').run_test()
    results.cap_ratio

Each pipeline executes exactly once: the load stores it pending, and
``run_test`` replays it (the ``rc_source`` side first) and consumes it. After
the run, ``tst.meas`` and ``tst.sim`` hold the same filtered data as the test
that produced the file, so the filtering summaries, visualizations, reporting
conditions, and capacity-ratio results match. The pipelines can also be run
one side at a time (``tst.run_test(side='meas')``) or manually with
:py:meth:`~captest.capdata.CapData.run_pipeline`, e.g.
``tst.meas.run_pipeline(tst.meas_filters_pending)``.

See :ref:`running-with-run-test` for the full description of ``run_test`` —
including how it chooses between an applied chain and a pending pipeline, and
re-running a single side after re-loading its data with
:py:meth:`~captest.captest.CapTest.reload` — and :ref:`captest-typical-workflow`
for the step-by-step states of a loaded test.

.. note::

    Replaying a pipeline with :py:meth:`~captest.capdata.CapData.run_pipeline`
    (which ``run_test`` uses internally) always resets the applied filter
    chain before rebuilding it — replay is restore-then-re-run. Appending a
    serialized pipeline onto an existing chain is not supported; extend a
    chain with the ``filter_*`` methods instead. Replay is also
    transactional: if a step fails, the chain and the reporting-conditions
    state are rolled back to their pre-call values, and a ``run_test`` replay
    failure retains the pipeline in ``<side>_filters_pending`` for editing
    (see :ref:`replay-failure`).

.. note::

    Filter arguments that depend on values computed during the test are encoded
    so they survive the round-trip. The narrower post-reporting-conditions
    irradiance filter, for example, is stored as ``ref_val: rep_irr`` rather than
    a hard-coded number, so it re-resolves against the reporting conditions
    recomputed by the replayed ``RepCond`` step.

The `Bifacial Capacity Test`_ example notebook runs a full bifacial test and
writes its configuration with ``to_yaml`` in the final section. The
`Capacity Test from Config`_ example notebook then re-runs that exact test from
the resulting ``bifi_config.yaml`` file and confirms the results are identical.

.. _Bifacial Capacity Test: ../examples/captest_class_bifi.html
.. _Capacity Test from Config: ../examples/captest_from_config.html
