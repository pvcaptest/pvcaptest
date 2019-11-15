# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]: https://github.com/pvcaptest/pvcaptest/compare/v0.6.0...HEAD
## [Unreleased]
### Added
- New filter_shade method separate from the filter_pvsyst method.

### Changed
- Filter_pvsyst method filters on IL Pmin, IL Pmax, IL Vmin, and IL Vmax and warns if any of the four are missing. Previously failed if any of the four were missing.

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
