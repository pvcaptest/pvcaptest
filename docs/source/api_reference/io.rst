.. currentmodule:: captest

Data Loading
============

Functions and classes for loading measured and simulated PV data into
:py:class:`~captest.capdata.CapData` objects.

Top-level Functions
-------------------

.. autosummary::
   :toctree: generated/

   io.load_data
   io.load_pvsyst

DataLoader
----------

:py:class:`~captest.io.DataLoader` handles multi-file loading, reindexing,
and joining before the data is passed to a :py:class:`~captest.capdata.CapData`
object.

.. autosummary::
   :toctree: generated/

   io.DataLoader

Methods
^^^^^^^

.. autosummary::
   :toctree: generated/

   io.DataLoader.load
   io.DataLoader.set_files_to_load
   io.DataLoader.reindex_loaded_files
   io.DataLoader.join_files
   io.DataLoader.sort_data
   io.DataLoader.drop_duplicate_rows
   io.DataLoader.reindex
