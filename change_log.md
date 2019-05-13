### v0.5.2
- Update name and location of conda environment yml file, so there is a single file and it works with binder.
- Removed binder directory.
- Update readme to reflect changes to conda environment.yml
- Minor updates to example.
- Minor documentation string updates.

### v0.5.1
- Changed Holoviews dependency to >= v1.11.  DatLink added in v1.11 is required for scatter_hv method.
- Expanded docstring for the load_data method to more clearly explain how the method joins multiple files (by row).
- Update installation directions in README.
- Updated conda environment file (conda_env.yml) to match updated dependencies.

### v0.5.0
- Moved all filtering and regression functionality from CapTest class into the CapData class and replace CapTest class with functions for results comparing CapData objects.
- Addition of clear sky modeling using pvlib library.  See new example notebook 'Clear Sky Examples'.
- Significant refactor of the rep\_cond function.  Removed any time filtering and prediction from rep\_cond.  Rep\_cond acts on filtered data in the df\_flt attribute.
- Added a new method, `predict_capacities` for calculating reporting conditions and predicted outputs by month.
- New example notebook demonstrating use of `rep_cond` and `predict_capacities`.
- `agg_sensors` method updated to be more explicit and flexible.
- Add warning when filter removes all data.
- Changed `filter_sensors` to filter based on percent difference between all combinations of pairs of sensors measuring the same environmental factor.  Corrected bug where standard deviation filter could not detect outliers with more than two, but still a small number of sensors.
- Adjusted bounds check of columns of data when importing so that translation dictionary names would not have 'valuesError' added to them.
- Made printout of bounds check results optional when loading data.
- Adjusted the type\_defs and sub\_type_defs, so that translation dictionary keys are more accurate for PVsyst data.