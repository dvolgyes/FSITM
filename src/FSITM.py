#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from contracts import contract

from .phasecong100 import phasecong100

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
