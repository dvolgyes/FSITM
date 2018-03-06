#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft.fftpack import fft2, ifft2
from numpy.fft import ifftshift
from contracts import contract
from lowpass_filter import lowpass_filter

# The orininal Matlab code is written by Peter Kovesi.
# This python code is a translation of his Matlab code,
# therefore, it is kind of a derived code.
# So, here is his original copyright notice:
#
# Copyright (c) 1996-2010 Peter Kovesi
# Centre for Exploration Targeting The University of Western Australia
# peter.kovesi@uwa.edu.au
#
# This function is optimized to generate one of the outputs of the 'phasecong3'
#  function, please see the original function at:
#  http://www.peterkovesi.com/matlabfns/
#
#  You can cite this publication:
#  @misc {KovesiMATLABCode,
#  author = "P. D. Kovesi",
#  title = "{MATLAB} and {Octave} Functions
#           for Computer Vision and Image Processing",
#  note = "Available from: $<$http://www.peterkovesi.com/matlabfns/$>$",
#  }
#
#  Reimplementation / translation:
#  If you want to contact me, e.g. you discovered a bug,
#  my name is David Völgyes, and you can write to here:
#   david.volgyes@ieee.org
#
#  I don't need credit for the reimplementation, you can use it as you want,
#  but as a derived product/code, you still should give credit to Peter Kovesi.
#


@contract(im='array(float)',
          nscale='int,>0',
          norient='int,>0',
          minWavelength='int,>1',
          sigmaOnf='float,>0')
def phasecong100(im, nscale=2,
                 norient=2,
                 minWavelength=7,
                 mult=2,
                 sigmaOnf=0.65):
    #
    #     im                       # Input image
    #     nscale          = 2;     # Number of wavelet scales.
    #     norient         = 2;     # Number of filter orientations.
    #     minWaveLength   = 7;     # Wavelength of smallest scale filter.
    #     mult            = 2;     # Scaling factor between successive filters.
    #     sigmaOnf        = 0.65;  # Ratio of the standard deviation of the
    #                              # Gaussian describing the log Gabor filter's
    #                              # transfer function in the frequency domain
    #                              # to the filter center frequency.

    rows, cols = im.shape
    imagefft = fft2(im)
    zero = np.zeros(shape=(rows, cols))

    EO = dict()
    EnergyV = np.zeros((rows, cols, 3))

    x_range = np.linspace(-0.5, 0.5, num=cols, endpoint=True)
    y_range = np.linspace(-0.5, 0.5, num=rows, endpoint=True)

    x, y = np.meshgrid(x_range, y_range)
    radius = np.sqrt(x ** 2 + y ** 2)

    theta = np.arctan2(- y, x)

    radius = ifftshift(radius)

    theta = ifftshift(theta)

    radius[0, 0] = 1.

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp = lowpass_filter((rows, cols), 0.45, 15)

    logGabor = []
    for s in range(1, nscale + 1):
        wavelength = minWavelength * mult ** (s - 1.)
        fo = 1.0 / wavelength
        logGabor.append(np.exp(
            (- (np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2)
        ))
        logGabor[-1] *= lp
        logGabor[-1][0, 0] = 0

    # The main loop...
    for o in range(1, norient + 1):
        angl = (o - 1.) * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * norient / 2., np.pi)
        spread = (np.cos(dtheta) + 1.) / 2.
        sumE_ThisOrient = zero.copy()
        sumO_ThisOrient = zero.copy()
        for s in range(0, nscale):
            filter_ = logGabor[s] * spread
            EO[(s, o)] = ifft2(imagefft * filter_)
            sumE_ThisOrient = sumE_ThisOrient + np.real(EO[(s, o)])
            sumO_ThisOrient = sumO_ThisOrient + np.imag(EO[(s, o)])
        EnergyV[:, :, 0] = EnergyV[:, :, 0] + sumE_ThisOrient
        EnergyV[:, :, 1] = EnergyV[:, :, 1] + np.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] = EnergyV[:, :, 2] + np.sin(angl) * sumO_ThisOrient
    OddV = np.sqrt(EnergyV[:, :, 0] ** 2 + EnergyV[:, :, 1] ** 2)
    featType = np.arctan2(EnergyV[:, :, 0], OddV)
    return featType
