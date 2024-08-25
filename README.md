# RAIT-Python
## Python implementation of RAIT library

Requires Pytorch >= 2.3 (likely runs on lower versions, though untested)

## Presentation

This repository implements the RAIT matlab library

## Known issues
### Overinstallation
`pip install -e .` will install far more pytorch-related packages than what is probably needed to run on desktop (not necessarily GPU). For now it is recommended to install pytorch from the official site.

Check setup.py if this can be remedied.
### Unimplemented/untested functions
Currently the following functions are not implemented/tested:

Functions that plot:
- arg_inv_anim
- blaschkes_img
- periodize
- rshow

Functions starting with simplex:
- simplex_biort
- simplex_biortdc
- simplex_mt
- simplex_mtdc
- simplex_mtdr

### License is also missing

## Installation

Type `pip install -e .` in the root folder of this repo.

## Testing

`pip install pytest`

`python -m pytest`




