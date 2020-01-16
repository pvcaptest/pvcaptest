# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]: https://github.com/pvcaptest/pvcaptest/compare/v0.6.0...HEAD
## [Unreleased]
### Added
- New filter_shade method separate from the filter_pvsyst method.
- cp_results method warns when it automatically attempts to correct for W vs kW.

### Changed
- Filter_pvsyst method filters on IL Pmin, IL Pmax, IL Vmin, and IL Vmax and warns if any of the four are missing. Previously failed if any of the four were missing.
- cp_results returns a warning if the regression formulas of the passed CapData objects do not match instead of warning and continuing.
- pvlib 0.6.3 is required; there are issues introduced by the pvlib 0.7.0 release not yet addressed.

Names were changed to remove unclear abbreviations:
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
- Examples moved from root/examples directory to docs/examples.
- Executed versions of the examples display on read the docs.
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
