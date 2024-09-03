# RAIT-Python
## Python implementation of RAIT library
Python >= 3.9 recommended, as there are type hints with tuples

Requires Pytorch >= 2.3 (likely runs on lower versions, though untested) - [how to install pytorch](https://pytorch.org/get-started/locally/)

Requires Spicy (for binom)

## Presentation

This repository implements the RAIT matlab library using Pytorch for GPU compatibility (instead of numpy)

## Known issues
### Manual installation of dependencies
`pip install -e .` will probably not be enough for proper use of this library. The dependencies need to be installed manually

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

### License copyright holder name is also missing

## Installation

Type `pip install -e .` in the root folder of this repo.

## Testing

`pip install pytest` if pytest is not installed

`python -m pytest` in the root folder of this repo.

## References
> [1] [Kovács, P., Lócsi, L., RAIT: The Rational Approximation and Interpolation Toolbox for MATLAB, Proceedings of the 35th International Conference on Telecommunication and Signal Processing (TSP), 2012, pp. 671-677.](http://dx.doi.org/10.11601/ijates.v1i2-3.18) 


## FAQ
### How to know the dependencies of the functions in the matlab version?

Install the 30-day free trial version of Matlab, and use the [dependency analyzer](https://www.mathworks.com/help/matlab/ref/dependencyanalyzer-app.html) function on the matlab version's folder. If you run out of time, you can always create a new account for another 30 days

## License
Copyright 2024 [copyright holder]

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE



