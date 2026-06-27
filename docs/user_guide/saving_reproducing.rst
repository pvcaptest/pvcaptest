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

Loading the file with :py:meth:`~captest.captest.CapTest.from_yaml` reproduces
the test in one step. It loads the measured and modeled data, runs
:py:meth:`~captest.captest.CapTest.setup` to assign the regression and calculated
columns, and then re-applies both filter pipelines in order:

.. code-block:: Python

    tst = CapTest.from_yaml('./project.yaml')

After this call ``tst.meas`` and ``tst.sim`` hold the same filtered data as the
original test, so the filtering summaries, visualizations, reporting conditions,
and capacity-ratio results match the run that produced the file.

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
