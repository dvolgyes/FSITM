#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from contracts import contract
import sys

from phasecong100 import phasecong100

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

    Q = np.sum(np.logical_and(PhaseL <= 0, PhaseH <= 0) +
               np.logical_and(PhaseL > 0, PhaseH > 0)) / NumPixels
    return Q


@contract(HDR='array[NxM](float)',
          LDR='array[NxM](float)',
          alpha='float,>=0.0,<=1.0')
def FSITM_revised(HDR, LDR, alpha):
    # HDR: High dynamic range image
    # LDR: Low dynamic range image
    # Q: Quality index

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
            metric = FSITMr
        else:
            metric = FSITM
        result = metric(hdr, ldr)
        prec = options.precision

        print("S: ",np.around(result,prec))
