.. currentmodule:: captest

Prep
====

The :py:mod:`captest.prep` module holds the data-preparation steps: declared,
serializable adjustments applied between loading data and
:py:meth:`~captest.captest.CapTest.setup`. Each step is a
:py:class:`~captest.prep.BasePrepStep` subclass that declares its
configuration as typed ``param`` parameters and implements ``_execute``,
rewriting ``CapData.data`` in place and recording itself on
``CapData.prep``. The ``prep_*`` methods on
:py:class:`~captest.capdata.CapData` are thin wrappers that build the matching
step and ``run()`` it. Steps may also be constructed and run directly against
a :py:class:`~captest.capdata.CapData` instance ``cd``:

.. code-block:: Python

   from captest.prep import ConvertUnits, Scale

   ConvertUnits(group_regex='^temp', from_units='F', to_units='C').run(cd)
   Scale(columns=['E_Grid'], factor=0.001).run(cd)

See :ref:`data_prep` in the user guide for the workflow.

Base Class
----------

:py:class:`~captest.prep.BasePrepStep` owns the ``run()`` lifecycle, the
three-way column selector (``columns`` / ``group`` / ``group_regex``), the
``custom_name`` label, and the guard that refuses a value-rewriting step once
filters have been applied.

.. autosummary::
   :toctree: generated/

   prep.BasePrepStep

Prep Steps
----------

Each step mutates ``CapData.data`` in place. The corresponding
``CapData.prep_*`` wrapper (or, for the last two,
:py:meth:`~captest.capdata.CapData.drop_cols` /
:py:meth:`~captest.capdata.CapData.rename_cols`) builds and runs the step.

.. autosummary::
   :toctree: generated/

   prep.ConvertUnits
   prep.Scale
   prep.AsType
   prep.DropColumns
   prep.RenameColumns
   prep.Custom

Unit Conversions
----------------

Conversions are affine (``out = value * factor + offset``). Only the forward
direction of each pair is tabulated in ``UNIT_CONVERSIONS``; inverses are
derived. ``UNIT_ALIASES`` normalizes unit spellings case-insensitively before
lookup.

.. autosummary::
   :toctree: generated/

   prep.conversion_factors

Serialization
-------------

``PREP_REGISTRY`` maps a step's type name to its class;
:py:func:`~captest.prep.prep_step_from_config` is the inverse of each step's
``to_config``.

.. autosummary::
   :toctree: generated/

   prep.prep_step_from_config
