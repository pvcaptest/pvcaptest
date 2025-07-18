# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]

[0.13.2]: https://github.com/pvcaptest/pvcaptest/compare/v0.13.1...v0.13.2
## [0.13.2] - 2025-07-17
### Fixed
- Change to nan for compatability with numpy 2.0
- Update file_reader to warn but load empty csv files

[0.13.1]: https://github.com/pvcaptest/pvcaptest/compare/v0.13.0...v0.13.1
## [0.13.1] - 2024-06-05
### Fixed
- Issue 112 - plotting tool would plot only last curve for plots with more than 25 curves.

[0.13.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.12.1...v0.13.0
## [0.13.0] - 2024-05-05
### Added
- `load_data` can read site location and system information for generating clear sky irradiance from a json or yaml file.
- New plotting module with a plot function which replaces the CapData.plot method. The new plot function creates a panel dashboard with expanded functionality. Internally, removes
the plotting dependency on the CapData.trans_keys attribute.
- `loc` and `floc` can be used to retrieve the regression columns, similar to previous rview functionality by using `regcols`.

### Changed
- Removed the CapData `trans_keys ` attribute, which was a copy of the `column_groups` keys and would be modified by the old `CapData.plot` method. `trans_keys` made it difficult to create a `CapData` object without using the `load_data` function.
- Replaces all uses of `view` and `rview` with `loc` and `floc`.
- Updated `loc` and `floc` to always return a DataFrame. Previously these would sometimes return a Series.
- `load_data` now checks if any individual files were loaded when loading multiple files from a directory.
- Adds underlay curve of unfiltered power to the linked timeseries created when calling `scatter_hv` with `timeseries=True`.
- Changes selected points on scatter plot and linked timeseries produced by `scatter_hv` to red.
- Tolerances may now be fractions eg '- 3.5'
- The plotting methods `scatter_hv`, `scatter_filters`, and `timeseries_filters` do not require a column labeled 'index' with string datetimes in the `data` DataFrame anymore. Also, the index of the `data` DataFrame does not need to be named 'Timeseries'.
- Removes the `add_index_col` kwarg option from `util.reindex`.

[0.12.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.11.2...v0.12.0
## [0.12.0] - 2023-08-27
### Added
- Added a user guide section to the documentation with an overview and bifacial tests section.
- Added verbose kwarg to the DataLoader load method, which prints which files have been loaded
- DataLoader load method has new kwarg print_errors
- DataLoader load method stores list of paths of files that fail to load in failed_to_load attribute
- CapData method to save column_groups dictionary to excel
- Github action to build and publish to PyPI on tags like v* pushed from master

### Changed
- Updates to make pvcaptest compatible with pvlib 0.10 and scipy 1.11
- Update to make pvcaptest compatible with bokeh v3.0.0, change `plot_width` and `plot_height` to `width` and `height`
- Make bokeh v3 minimum version
- Drop support for python 3.7
- Drop to_numeric from io.file_reader
- CapData.column_groups is instance of ColumnGroups now when group_columns is a callable
- Changed defaults for the CapData plot method to single column

[0.11.2]: https://github.com/pvcaptest/pvcaptest/compare/v0.11.1...v0.11.2
## [0.11.2] - 2023-04-20
### Added
- Adds CI testing across python 3.7 to 3.11 on OSX, Linux, and Windows using Github actions.

### Changed
- Re-organized project root directory to place captest package under a src directory, see PR #83.
- Re-organized the extras for pip installation into optional, test, docs, and all. Updated the installation section of the docs to reflect change.
- Updated installation instructions with links to pandas and conda docs for more information on environments.
- Re-organized installation instructions.
- Changed instructions on creating a conda env for pvcaptest to directly download the env yaml file from the repository rather than the whole repository.
- Clean up RTD configuration, particularly added project directory to PYTHONPATH in the docs build environment, so the docs build against checked out version instead of installed version from conda.
- Updates to documentation to add new modules, remove history section of releases, and remove references to Travis CI.

### Fixed
- Cleaned up issues in tests found after re-implementing CI on other platforms.

[0.11.1]: https://github.com/pvcaptest/pvcaptest/compare/v0.11.0...v0.11.1
## [0.11.1] - 2023-04-09
### Added
- Added new dependencies - colorcet, param
- Added openpyxl as an optional dependency
- Loads bokeh as holoviews extension in captest imports so user doesn't need to

### Changed
- Removed use of hvplot, replaced with holoviews to avoid adding unnecessary dependency. PR #73.

### Fixed
- Typo in the import check of panel

[0.11.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.10.0...v0.11.0
## [0.11.0] - 2023-04-07 [YANKED]
### Added
- Added columngroups module with a ColumnGroups class that extends python
dictionaries to include each column group as an attribute and __repr__ is
formatted for easy reading.
- Created the `loc` and `floc` callables for CapData, which allow easier access to columns of the `data` and `data_filtered` attributes, repsectively. Will replace `view` and `rview` method.
- Added `read_json` function to util module.
- Added `read_yaml` function to util module.
- Added the io module with functions to load data and return instances of CapData containing the loaded data.
- Added the ReportingIrradiance class for calculations of a reporting irradiance that is between the 40th and 60th percentile. `ReportingIrradiance.plot` includes a dashboard type view of the selected and possible reporting irradiance values and method to save possible reporting conditions table to csv.
- Added the `spatial_uncert` and `expanded_uncert` methods. Improvements and testing needed.


### Changed
- Moved group_columns method, series_type, and type definitions from the capdata module to the columngroups module and changed group_columns from a CapData method to a function that returns a ColumnGroups instance.
- Completely refactored the algorithim to determine the reporting irradiance when `irr_bal` is set to True. Now an instance of the new ReportingIrradiance class is created and used to determine the reporting irradiance.
- Updated the examples to reflect the changes to the API for data loading, column grouping, selection (loc and floc vs view and rview), and balanced reporting irradiance.
- Changed the `scatter_hv` method to be more flexible and not require temperature or wind regression columns.

### Removed
- Removed the value checking functionality of the group_columns function. Data quality is outside the scope of the pvcaptest project.
- Removed the `load_data`, `load_das`, and `load_pvsyst` methods from the CapData class. Replaced with io module.



[0.10.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.9.0...v0.10.0
## [0.10.0] - 2021-07-25
### Added
- Added the filter_missing CapData method to remove missing data from specified columns.
By default removes only intervals that contain missing data in the regression variable
columns.
- Added option to filter_irr method to specify using the reporting irradiance in the CapData object as the reference irradiance.
- Added option to the filter_time method to drop the specified time period instead of dropping all other times.
- Added option to filter_clearsky method to keep time periods with unstable irradiance.
- Added new attributes to CapData: removed, kept, filter_counts. The update_summary decorator now stores the
index of points removed, the index of points remaining after each filter, and the number of times any filter has been run for each filter applied.
- Adds new plotting method, scatter_filters, which shows which filtering step removed which time intervals of data in a plot of irradiance vs. power.
- New plotting method, timeseries_filters, which shows which fitlering step removed which time intervals of data in a plot of power vs. time.
- New plotting function, overlay_scatters, that overlays irradiance vs. power scatter plots of the data remaining after the last filtering step of the two CapData objects passed to the function.
- New get_filtering_table method that returns a DataFrame documenting the which time intervals are removed by which filter and which time intervals remain after all filtering.
- Adds the run_test function, which applies the passed list of CapData filtering methods to the CapData object passed.
- Adds the points_summary method, which prints the number of points remaining after all filtering, the length of the test period, the average points remaining after filtering per day, if enough points have been collected, if
more points are needed how many, and how many days left if the rate of points holds.

### Changed
- Updated filter_pvsyst method to handle inverter output variables that have underscores
or spaces like 'IL Pmin' and 'IL_Pmin'.
- load_das method no longer drops columns and rows that contain no data
- Format of hover tooltip in plots produced by plot method now includes comma separator for thousands.
- Changes captest to pvcaptest in documentation.
- get_reg_cols method default changed to get and rename the columns defined in the `regression_cols` attribute
rather than expecting regression variables/columns to be identified by the keys 'power', 'poa', 't_amb', and 'w_vel' in the `regression_cols` attribute.


[0.9.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.8.0...v0.9.0
## [0.9.0] - 2020-08-16
### Changed
- Updated clear sky functions which rely on pandas `index.tz_localize` to use nonexistent argument rather than errors argument, which was deprecated in pandas v1.0.
- Made Pandas v1.0 or greater a requirement for pvcaptest.
- Change to test against python v3.7* and v3.8*
- Updated installation instructions.


[0.8.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.7.0...v0.8.0
## [0.8.0] - 2020-04-13
### Added
- Added a filter_power method to the CapData class.
- Added a filter_days method to the CapData class.

### Changed
- Allow get_reg_cols to accept a single regression variable as a string. Previously required passing list with at least two entries.
- Fixed bug in filter_clearksy that applied filter to data rather than data_filtered attribute.
- Added option to plot method to use column names for hover labels instead of abbreviated column names.
- Improved formatting of the filtering summary output. See issue #12 for details.
- Cleaned up source code by correcting linter errors.

[0.7.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.6.0...v0.7.0
## [0.7.0] - 2020-03-08
### Added
- New filter_shade method separate from the filter_pvsyst method.
- captest_results method warns when it automatically attempts to correct for W vs kW.

### Changed
- Filter_pvsyst method filters on IL Pmin, IL Pmax, IL Vmin, and IL Vmax and warns if any of the four are missing. Previously failed if any of the four were missing.
- cp_results returns a warning if the regression formulas of the passed CapData objects do not match instead of warning and continuing.
- Updates to make captest compatible with pvlib 0.7.0
- Editing of the complete capacity test example to use new names and improve explanations of features.

Names were changed to remove ambiguous abbreviations:
- flt - filter; API changes in many places
- cntg_eoy - wrap_year_end; API change
- cp_results - captest_results; API change
- res_summary - captest_results_check_pvalues; API change
- reg_fml - regression_formula; API change
- irrRC_balanced - irr_rc_balanced; API change
- df_beg - df_start
- ix_ser - ix_series
- mnth - month
- months_boy - months_year_start
- months_eoy - months_year_end
- loop_cnt - loop_count
- cprat - cap_ratio
- cprat_cpval - cap_ratio_check_pvalues
- trans - column_groups; API change
- set_translation - group_columns
- trans_report - column_type_report
- set_trans argument of load_data - group_columns
- review_trans - review_column_groups
- set_reg_trans - set_regression_cols
- reg_trans - regression_cols
- update_reg_trans argument of agg_sensors - update_regression_cols
- reg_cpt - fit_regression
- ols_model - regression_results

### Removed
- Removed the inv_trans_dict function. This was intended for use within the module and was unused.

[0.6.0]: https://github.com/pvcaptest/pvcaptest/compare/v0.5.3...v0.6.0
## [0.6.0] - 2019-09-15
### Added
- Setup Travis CI to test pull requests and test and deploy to pypi for tags on master.
- Setup project documentation hosted by Read the Docs using sphinx, nbshpinx, napolean, recommonmark, AutoStructify

### Changed
- Versioning changed from manual update in __version.py file to using versioneer to update version number from git tag.
- Updated this file to follow the Keep a Changelog formatting conventions.
- Moved repository to an organization github account from my personal github account.
- Examples moved from root/examples directory to docs/examples. Executed versions of the examples display on read the docs.
- All examples can be launched through binder in live notebooks.
- The environment file has been updated to work for binder and Read the Docs.

[0.5.3]: https://github.com/pvcaptest/pvcaptest/compare/v0.5.1...v0.5.3
## [0.5.3] - 2019-05-12
### Changed
- Update name and location of conda environment yml file, so there is a single file and it works with binder.
- Removed binder directory.
- Update readme to reflect changes to conda environment.yml
- Minor updates to example.
- Minor documentation string updates.

[0.5.1]: https://github.com/pvcaptest/pvcaptest/compare/v0.4.0...v0.5.1
## [0.5.1] - 2019-05-01
### Added
- Addition of clear sky modeling using pvlib library.  See new example notebook 'Clear Sky Examples'.
- Added a new method, `predict_capacities` for calculating reporting conditions and predicted outputs by month.
- New example notebook demonstrating use of `rep_cond` and `predict_capacities`.
- Add warning when filter removes all data.

### Changed
- Changed Holoviews dependency to >= v1.11.  DatLink added in v1.11 is required for scatter_hv method.
- Expanded docstring for the load_data method to more clearly explain how the method joins multiple files (by row).
- Update installation directions in README.
- Updated conda environment file (conda_env.yml) to match updated dependencies.
- **Moved all filtering and regression functionality from CapTest class into the CapData class and replace CapTest class with functions for results comparing CapData objects.**
- **Significant refactor of the rep\_cond function.  Removed any time filtering and prediction from rep\_cond.  Rep\_cond acts on filtered data in the df\_flt attribute.**
- `agg_sensors` method updated to be more explicit and flexible.
- Changed `filter_sensors` to filter based on percent difference between all combinations of pairs of sensors measuring the same environmental factor.  Corrected bug where standard deviation filter could not detect outliers with more than two, but still a small number of sensors.
- Adjusted bounds check of columns of data when importing so that translation dictionary names would not have 'valuesError' added to them.
- Made printout of bounds check results optional when loading data.
- Adjusted the type\_defs and sub\_type_defs, so that translation dictionary keys are more accurate for PVsyst data.
