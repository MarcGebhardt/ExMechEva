# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.1.2] – 2024-02-20

### Added

- Added least square regression fit functionality (./common/fitting.py; used in conclusion evaluation)
	- Genaral functions with equation string builders (linear, power law and exponential)
	- Variable fitting functionality (./common/fitting.regfitret)
	- Wrapping functionality for multiple combination (./common/stat_ext.reg_stats_multi)
	- Plotting functionality on matplotlib axis level (./common/plotting.plt_ax_regfit)
- Test of elastic modulus determination by generic three-point-bending test data reimplemented (./tests/test_tbt_emdet.py)

### Changed

- Documentation improved
- Conclusion evaluation improved (./scripts/01_PF/Eva_Conclusion_PMat.py)
- Spring force reduction made dependent on the number of displacement measurement devices (LVDTs, only relevant for axial tensile test)

## [v0.1.1] – 2024-01-25

### Added

- Packing of log-files of series (common.loadnsave.comb_logs)
- Identifying type of given object (standard, numpy and pandas; common.helper.type_str_return)
- Relative deviation implemented (common.stat_ext.relative_deviation)
- Coefficient of variation (with or without statistical outliers) with short named function (common.stat_ext.cv and common.stat_ext.cvwoso) for use in pandas aggregate
- Functionality for automatic determination of points of interest added to ./common/mc_char (poi_rel_finder, poi_det_plh, poi_vip_namer, poi_fixeva and poi_refinement)

### Changed

- Documentation improved
- Packing of log-files added in common.eva.selector
- common.stat_ext: 
	- moved imports to start of file
	- coefficient_of_variation and coefficient_of_variation_woso using absolute value of mean by default (preventing CV to be negative)
	
### Removed

- Special evaluation of three-point-bending tests for Young's modulus determination paper (./scripts/01_PF/Methods_Verif/, old structure, https://doi.org/10.1007/s11340-023-00945-y)
- Special evaluation of preparation, geometry and assessment codes for standardized preparation paper (./scripts/01_PF/Eva_Preparation.py, old structure, https://doi.org/10.1371/journal.pone.0289482)
- Evaluation of three-point-bending tests of cortical bone specimens with different moisture states (./scripts/02_MMCB/, old structure), Including different improvements:
	- improvements of measured curve characterization (MCurve_Characterizer, MCurve_Char_Plotter,...)
	- automatic setting of load direction by optical measured traverse displacement (ldoption='auto-Pcoorddisp' in bending/fitting/Perform_Fit)
	- automatic determination and naming of points of interest (see ./common/mc_char (poi_rel_finder, poi_det_plh, poi_vip_namer, poi_fixeva and poi_refinement)
- Mandibulae evaluation (./scripts/03_Zesbo_Mand/, old structure, no relevant changes to other projects)

## [v0.1.0] – 2024-01-17

### Added

- First version with structure, full documentation and packaging


[unreleased]: https://github.com/MarcGebhardt/ExMechEva/tree/main
[v0.1.2]: https://github.com/MarcGebhardt/ExMechEva/releases/tag/v0.1.2
[v0.1.1]: https://github.com/MarcGebhardt/ExMechEva/releases/tag/v0.1.1
[v0.1.0]: https://github.com/MarcGebhardt/ExMechEva/releases/tag/v0.1.0