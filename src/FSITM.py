#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
from numpy.fft.fftpack import fft2, ifft2
from numpy.fft import ifftshift
from contracts import contract
from functools import partial

if not (sys.version_info >= (3, 5)):
    sys.stdout.write("Sorry, requires Python 3.5 or newer.\n")
    sys.exit(1)


#  Feature similarity index for tone mapped images (FSITM)
#  By: Hossein Ziaei Nafchi, November 2014
#  hossein.zi@synchromedia.ca
#  Synchromedia Lab, ETS, Canada

#  The code can be modified, rewritten and used without obtaining permission
#  of the authors.

#  Please refer to the following paper:
#  FSITM: A Feature Similarity Index For Tone-Mapped Images
#  Hossein Ziaei Nafchi, Atena Shahkolaei,
#      Reza Farrahi Moghaddam, Mohamed Cheriet,
#  IEEE Signal Processing Letters, vol. 22, no. 8, pp. 1026-1029, 2015.
#  DOI:  10.1109/LSP.2014.2381458

#  Needs phasecong100 function (which requires lowpassfilter function)
#  implemented later


@contract(HDR='array[NxM](float)', LDR='array[NxM](float)')
def FSITM(HDR, LDR, alpha = None):
    """
    HDR: High dynamic range image
    LDR: Low dynamic range image

    The original implementation used color channels and
    selected a single channel for processing. This makes limited
    sense, so this reimplementation requires a single channel.
    If any color conversion is needed, it should be performed
    before this function is called.

    """

    # HDR: High dynamic range image
    # LDR: Low dynamic range image
    # Q: Quality index
    NumPixels = LDR.size

    if alpha is None:
        r = np.floor(NumPixels / (2. ** 18))
        if r > 1.:
            alpha = 1. - (1. / r)
        else:
            alpha = 0.

    minNonzero = np.min(HDR[HDR > 0])
    LogH = np.log(np.maximum(HDR, minNonzero))

    # float is needed for further calculation
    LogH = np.round((LogH - LogH.min()) * 255. /
                    (LogH.max() - LogH.min())).astype(np.float)

    if alpha > 0.:
        PhaseHDR_CH = phasecong100(HDR, 2, 2, 8, 8)
        PhaseLDR_CH8 = phasecong100(LDR, 2, 2, 8, 8)
    else:  # so, if image size is smaller than 512x512?
        PhaseHDR_CH = 0
        PhaseLDR_CH8 = 0

    PhaseLogH = phasecong100(LogH, 2, 2, 2, 2)
    PhaseH = alpha * PhaseHDR_CH + (1 - alpha) * PhaseLogH

    PhaseLDR_CH2 = phasecong100(LDR, 2, 2, 2, 2)
    PhaseL = alpha * PhaseLDR_CH8 + (1 - alpha) * PhaseLDR_CH2
    Q = np.sum(np.logical_or(np.logical_and(PhaseL <= 0, PhaseH <= 0),
               np.logical_and(PhaseL > 0, PhaseH > 0))) / NumPixels
    return Q


@contract(HDR='array[NxM](float)',
          LDR='array[NxM](float)',
          alpha='float,>=0.0,<=1.0|None')
def FSITMr(HDR, LDR, alpha):
    # HDR: High dynamic range image
    # LDR: Low dynamic range image
    # Q: Quality index

    if alpha is None:
        r = np.floor(HDR.size / (2. ** 18))
        if r > 1.:
            alpha = 1. - (1. / r)
        else:
            alpha = 0.

    LogH = HDR - HDR.min()
    minNonzero = np.min(LogH[LogH > 0])
    LogH = LogH + minNonzero
    LogH = np.log(LogH)

    if alpha > 0.:
        PhaseHDR_CH = phasecong100(HDR, 2, 2, 8, 8)
        PhaseLDR_CH8 = phasecong100(LDR, 2, 2, 8, 8)
    else:
        PhaseHDR_CH = 0
        PhaseLDR_CH8 = 0

    PhaseLogH = phasecong100(LogH, 2, 2, 2, 2)
    PhaseH = alpha * PhaseHDR_CH + (1 - alpha) * PhaseLogH

    PhaseLDR_CH2 = phasecong100(LDR, 2, 2, 2, 2)
    PhaseL = alpha * PhaseLDR_CH8 + (1 - alpha) * PhaseLDR_CH2

    # The original implementation used a <=0, >0 distinction, but
    # wouldn't make comparing the sign more sense?
    Q = np.sum(np.sign(PhaseL) == np.sign(PhaseH)) / LDR.size

    # Q = np.sum(np.logical_and(PhaseL <= 0, PhaseH <= 0) +
    #    np.logical_and(PhaseL > 0, PhaseH > 0)) / NumPixels
    return Q


def img_read(link, gray=False, shape=None, dtype=None, keep=False):
    if os.path.exists(link):
        if dtype is None:
            img = imread(link)
            if gray and len(img.shape) > 2:
                img = skimage.color.rgb2hsv(img)[..., 2]
        else:
            W, H = shape
            img = np.fromfile(link, dtype=dtype)
            if gray:
                img = img.reshape(H, W)
            else:
                img = img.reshape(H, W, -1)
    else:
        tempfile = wget.download(link, bar=None)
        img = img_read(tempfile, gray, shape, dtype)
        if not keep:
            os.remove(tempfile)
    return img.astype(np.float)



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
    if len(sys.argv) == 1:
        import doctest
        doctest.testmod()

    if len(sys.argv) > 1:  # there are command line parameters
        # these imports are unnecessary if the code is used as a library
        from optparse import OptionParser
        from scipy.misc import imsave
        from imageio import imread
        import os.path
        import wget
        import skimage.color

        usage = ("usage: %prog [options] HDR_image LDR_image\n" +
                 "The images could be files or a http(s)/ftp link.")
        parser = OptionParser(usage=usage)

        parser.add_option("-t", "--type",
                          type="string",
                          dest="maptype",
                          help="s_map file type (default: float32)",
                          default="float32")


        parser.add_option("-p", "--precision",
                          type="int",
                          dest="precision",
                          help="precision (number of decimals) (default: 4)",
                          default=4)

        parser.add_option("-W", "--width",
                          type="int",
                          dest="width",
                          help="image width (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-H", "--height",
                          type="int",
                          dest="height",
                          help="image height (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-i", "--input_type",
                          type="string",
                          dest="input",
                          help="type of the input images: float32/float64"
                          " for RAW images\n"
                          "None for regular images opening with scipy"
                          " (e.g. png) (default: None)",
                          default=None)

        parser.add_option("-g", "--gray",
                          dest="gray",
                          action="store_true",
                          help="gray input (ligthness/brightness)"
                          "  (default: RGB)",
                          default=False)

        parser.add_option("-Q", "--report-Q",
                          dest="report_q",
                          action="store_true",
                          help="report quality index",
                          default=True)

        parser.add_option("-C", "--report-channels",
                          dest="report_c",
                          action="store_true",
                          help="report structural similarity",
                          default=False)

        parser.add_option("-q", "--no-report-Q",
                          dest="report_q",
                          action="store_false",
                          help="do not report quality index")

        parser.add_option("-c", "--no-report-channels",
                          dest="report_s",
                          action="store_false",
                          help="do not report structural similarity")

        parser.add_option("--quiet",
                          dest="quiet",
                          action="store_true",
                          help="suppress variable names in the report")

        parser.add_option("--verbose",
                          dest="quiet",
                          action="store_false",
                          help="use variable names in the report (default)",
                          default=False)

        parser.add_option("--keep",
                          dest="keep",
                          action="store_true",
                          help="keep downloaded files (default: False)",
                          default=False)

        parser.add_option("-r","--revised",
                          dest="revised",
                          action="store_true",
                          help="Enable revised TMQI. (default: Original)")

        parser.add_option("-a","--alpha",
                          dest="alpha",
                          type="float",
                          action="store",
                          help="Alpha value. (default: original implementation)",
                          default=None)

        (options, args) = parser.parse_args()

        if len(args) != 2:
            print("Exactly two input files are needed: HDR and LDR.")
            sys.exit(0)

        if options.input is not None:
            W, H = options.width, options.height
            shape = (W, H)
            dtype = np.dtype(options.input)
        else:
            shape = None
            dtype = None

        hdr = img_read(args[0], True, shape, dtype, options.keep)
        ldr = img_read(args[1], True, shape, dtype, options.keep)

        if options.revised:
            metric = partial(FSITMr,alpha=options.alpha)
        else:
            metric = FSITM
        result = metric(hdr, ldr)
        prec = options.precision

        print("S: %s" % result)
