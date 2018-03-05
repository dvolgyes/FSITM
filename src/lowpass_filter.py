#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import ifftshift
from contracts import contract

# The orininal Matlab code is written by Peter Kovesi.
# This python code is a translation of his Matlab code,
# therefore, it is kind of a derived code.
# So, here is his original copyright notice and usage comment
# (with a variable renaming from 'sze' to 'size'):

#  LOWPASSFILTER - Constructs a low-pass butterworth filter.
#
#  usage: f = Lowpassfilter(size, cutoff, n)
#
#  where: size    is a two element vector specifying the size of filter
#                to construct [rows cols].
#         cutoff is the cutoff frequency of the filter 0 - 0.5
#         n      is the order of the filter, the higher n is the sharper
#                the transition is. (n must be an integer >= 1).
#                Note that n is doubled so that it is always an even integer.
#
#                       1
#       f =    --------------------
#                               2n
#               1.0 + (w/cutoff)
#
#  The frequency origin of the returned filter is at the corners.
#
#  See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER
#

# Copyright (c) 1999 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# http://www.csse.uwa.edu.au/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.

# October 1999
# August  2005 - Fixed up frequency ranges for odd and even sized filters
#                (previous code was a bit approximate)


# Reimplementation / translation:
# If you want to contact me, my name is David Völgyes,
# and you can write to here: david.volgyes@ieee.org
#
# I don't need credit for the reimplementation, you can use it as you want,
# but as a derived product/code, you still should give credit to Peter Kovesi.
#
# You can cite his publication:
# @misc {KovesiMATLABCode,
# author = "P. D. Kovesi",
# title = "{MATLAB} and {Octave} Functions
#          for Computer Vision and Image Processing",
# note = "Available from: $<$http://www.peterkovesi.com/matlabfns/$>$",
# }

@contract(size='int|tuple[2]',
          cutoff='float,>0,<=0.5',
          n='int,>=1')
def lowpass_filter(size=None, cutoff=None, n=None):
    """ Butterworth filter

    Examples:
    >>> lowpass_filter(3,0.5,2)
    array([[1. , 0.5, 0.5],
           [0.5, 0.2, 0.2],
           [0.5, 0.2, 0.2]])
    """
    if type(size) == int:
        rows = cols = size
    else:
        rows, cols = size
    x_range = np.linspace(-0.5, 0.5, num=cols)
    y_range = np.linspace(-0.5, 0.5, num=rows)
    x, y = np.meshgrid(x_range, y_range)
    radius = np.sqrt(x ** 2 + y ** 2)
    f = ifftshift(1.0 / (1.0 + (radius / cutoff) ** (2 * n)))
    return f


if __name__ == "__main__":
    import doctest
    doctest.testmod()
