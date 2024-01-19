# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Packing of log-files of series (common.loadnsave.comb_logs)
- Identifying type of given object (standard, numpy and pandas; common.helper.type_str_return)
- Relative deviation implemented (common.stat_ext.relative_deviation)
- Coefficient of variation (with or without statistical outliers) with short named function (common.stat_ext.cv and common.stat_ext.cvwoso) for use in pandas aggregate

### Changed

- Documentation improved
- Packing of log-files added in common.eva.selector
- common.stat_ext: 
	- moved imports to start of file
	- coefficient_of_variation and coefficient_of_variation_woso using absolute value of mean by default (preventing CV to be negative)

## [v0.1.0] – 2023-01-17

### Added

- First version with structure, full documentation and packaging


[unreleased]: https://github.com/MarcGebhardt/ExMechEva/tree/main
[v0.1.0]: https://github.com/MarcGebhardt/ExMechEva/releases/tag/v0.1.0