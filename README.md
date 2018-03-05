Feature similarity index for tone mapped images (FSITM) - revised
=================================================================

Travis CI: [![Build Status](https://travis-ci.org/dvolgyes/FSITM.svg?branch=master)](https://travis-ci.org/dvolgyes/FSITM)
Semaphore: [![Build Status](https://semaphoreci.com/api/v1/dvolgyes/FSITM/branches/master/badge.svg)](https://semaphoreci.com/dvolgyes/fsitm)
CircleCI: [![CircleCI](https://circleci.com/gh/dvolgyes/FSITM.svg?style=svg)](https://circleci.com/gh/dvolgyes/FSITM)

Coveralls: [![Coverage Status](https://img.shields.io/coveralls/github/dvolgyes/FSITM/master.svg)](https://coveralls.io/github/dvolgyes/FSITM?branch=master)
Codecov: [![codecov](https://codecov.io/gh/dvolgyes/FSITM/branch/master/graph/badge.svg)](https://codecov.io/gh/dvolgyes/FSITM)

This is a Python2/3 reimplementation of the Feature similarity index for tone mapped images (FSITM).

This implementation and the Matlab original have significant differences
and they yield different results!

The original article can be found here: http://ieeexplore.ieee.org/document/6985727/

DOI: [10.1109/LSP.2014.2381458](https://doi.org/10.1109/LSP.2014.2381458)

The reference implementation in Matlab: 
https://se.mathworks.com/matlabcentral/fileexchange/59814-fsitm-hdr--ldr--ch-

The original source code does not specify license, except that the code should be referenced
and the original paper should be cited.
I put this re-implementation under AGPLv3 license, hopefully this is compatible
with the original intention.

However, note that both the original and this  implementation build on the 
Matlab functions (lowpassfilter, phasecong100) from Peter Kovesi, see in the source code.
His function can be found here: http://www.peterkovesi.com/matlabfns/

Deviations
----------

I disagree with some implementation choices from the original article, e.g.

There are two functions implemented, FSITM which follows the original matlab code,
and FSITM_revised, which is slightly modified. The differences are:

- The HDR image is converted to uint8 before phase calculation.
  This might introduce a strong quantization error, and I see no reason for it.
  Leaving this rounding out, one of my tests jumped from .82 to 0.95.

- the alpha parameter is determined by the code, and I see no explanation why 
  was it chosen that way. The revised implementation asks alpha as an explicit
  parameter

Deviations in both implementations:

- Instead of selecting one of the  color channels, the codes require a single
  channel input.


These leads to different FSITM scores in the revised version, so the values from the original articles
and from this implementation are NOT comparable. Be careful before you choose one of them.

Install
-------

```
pip install git+https://github.com/dvolgyes/FSITM
```

Afterwards, you can import it as a library:
```
from FSITM import FSITM, FSITM_revised
```
