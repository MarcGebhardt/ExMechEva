# ExMechEva - Experimental mechanics evaluation

Evaluation of data determined by means of experimental mechanics.
Automation and standardization, considering special requirements of project and test specimen.

## Installation
`ExMechEva` is developed under Python 3.7.10 and is available in the [Python Package Index (PyPI)](https://pypi.org/project/ExMechEva/).

To install the latest stable version, please run:  
- Linux and Mac: `python3 -m pip install -U ExMechEva`
- Windows: `py -m pip install -U ExMechEva`   

To install the development version:  
- download/clone [Github](https://github.com/MarcGebhardt/ExMechEva)
- install requirements
- make the modules available by adding the `ExMechEva` directory to the `$PYTHONPATH` system variable or inside python with:  
```
import sys
sys.path.insert(-1,'Path\To\Project\ExMechEva')
```

## Getting started
In [EXAMPLES](./scripts/00_examples) you can find example scripts evaluating test data provided in [DATA](./data/test).
Available are:
- Simple axial compression test [ACT](./scripts/00_examples/ACT_Test.py) evaluating [ACT-DATA](./data/Test/ACT/Series_Test/)
- Cyclic preloaded axial tensile test [ATT](./scripts/00_examples/ATT_Test.py) evaluating [ATT-DATA](./data/Test/ATT/Series_Test/)
- Three-point bending test [TBT](./scripts/00_examples/TBT_Test.py), with different elastic modulus evalutaion types, evaluating [TBT-DATA](./data/Test/TBT/Series_Test/)

To start, select evaluation option (by uncommenting):
- 'single': Evaluate single measurement
- 'series': Evaluate series of measurements
- 'complete': Evaluate series of series
- 'pack': Pack all evaluations into single hdf-file (only results and evaluated measurement)
- 'pack-all': Pack all evaluations into single hdf-file with (all results, Warning: high memory requirements!)

## Contributing
See [CONTRIBUTING](./CONTRIBUTING.md) for details on how to contribute to `exmecheva`.
Please observe the [CODE OF CONDUCT](./CODE_OF_CONDUCT.md).
Overview of news is given in [CHANGELOG](./CHANGELOG.md).

## Development targets and ToDos
It is planned to extend the use of this package and make it as easily available as possible for scientific purposes.
Some important points should therefore be taken into account, which are for example:
- Improvement of documentation
- Implementation of test routines in [TESTS](./tests/)
- [PEP8](https://peps.python.org/pep-0008/) styleguide adaptations
- Updating of environment (now python 3.7.10)
- Development of a generally customizable routine that can be controlled with keywords. 
  (please see curve_characterizer and following in [Measurement Curve Characteriszation](./exmecheva/common/mc_char.py))
- Implementing of physical units ([pint](https://github.com/hgrecco/pint)/[pint-pandas](https://github.com/hgrecco/pint-pandas))
- GUI development for easy, reliable and reproducable use
- Implementing of option for input definition (p.e. positve direction of displacment measurement)
- Expandability to personal usage scenarios

## Citation
If you use this framework, please cite this paper:
```
@article{GebhardtKurz_2024_ASCTE,
  author          = {Marc Gebhardt, Sascha Kurz, Fanny Grundmann, Thomas Klink, Volker Slowik, Christoph-Eckhard Heyde, Hanno Steinke},
  date            = {planned 2024},
  journaltitle    = {Plos One},
  title           = {Approach to standardized material characterization of the human lumbopelvic system – testing and evaluation},
  doi             = {...},
  issn            = {1932-6203},
  language        = {english},
}
```
## Licence and Copyright
**Author:** Marc Gebhardt.  
**Copyright:** Copyright by the authors, 2023.  
**License:** This software is released under MIT licence, see [LICENSE](./LICENSE) for details.