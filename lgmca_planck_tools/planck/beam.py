#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
from .constants import planck_freqs
import fitsio
import numpy as np
import os

# FIXME: should these files be hard-coded or passed in config?
rimos_dir = os.path.join(os.path.dirname(__file__),
                         '..', 'data', 'ffp8', 'rimos')
lfi_data_file = os.path.join(rimos_dir, 'LFI_RIMO_R2.50.fits')
hfi_data_file = os.path.join(rimos_dir, 'HFI_RIMO_Beams-100pc_R2.00.fits')

LFI_BEAM_FREQ_NAMES = ['BEAMWF_030x030', 'BEAMWF_044X044', 'BEAMWF_070X070']
HFI_BEAM_FREQ_NAMES = ['BEAMWF_100x100', 'BEAMWF_143X143', 'BEAMWF_217X217', 'BEAMWF_353X353', 'BEAMWF_545X545', 'BEAMWF_857X857']

# Load in R2 - the beam for each channel.
r2_channel_beams = np.zeros((9, 4001))
with fitsio.FITS(lfi_data_file) as lfi_data, fitsio.FITS(hfi_data_file) as hfi_data:
    for i in range(9):
        if i < 3:
            beam = lfi_data[LFI_BEAM_FREQ_NAMES[i]]['Bl'].read()
        else:
            beam = hfi_data[HFI_BEAM_FREQ_NAMES[i-3]]['nominal'].read()
        r2_channel_beams[i, 0:beam.size] = beam


r3_dir = os.path.join(os.path.dirname(__file__),
                      '..', 'data', 'rimos')
r3_channel_beams = np.zeros((9, 4001))
with fitsio.FITS(os.path.join(r3_dir, 'LFI_RIMO_R3.31.fits')) as lfi_data:
    for i in range(3):
        bwf_hdr = 'beamwf_{freq:03}x{freq:03}'.format(freq=planck_freqs[i])
        bwf = lfi_data[bwf_hdr]['bl'].read()
        r3_channel_beams[i, :bwf.size] = bwf

for i in range(3, 9):
    if i < 7:
        fname = 'Wl_R3.01_fullsky_{freq:03}x{freq:03}.fits'.format(freq=planck_freqs[i])
    else:
        fname = 'Bl_T_R3.01_fullsky_{freq:03}x{freq:03}.fits'.format(freq=planck_freqs[i])
    fpath = os.path.join(r3_dir, 'BeamWf_HFI_R3.01', fname)
    with fitsio.FITS(fpath) as hfi_data:
        if i < 7:
            bwf = np.sqrt(hfi_data['TT']['TT_2_TT'].read()[0])
        else:
            bwf = hfi_data['window function']['temperature'].read()
        r3_channel_beams[i, :bwf.size] = bwf

r2_channel_beams.setflags(write=False)
r3_channel_beams.setflags(write=False)


beamwf_dir = os.path.join(os.path.dirname(__file__),
                          '..', 'data', 'rimos', 'BeamWf_HFI_R3.01')


def load_wl_freq(chan, masktype='fullsky', maptype='', lmax=4000):
    '''
    Loads :math:`W_\ell^{X, Y}` for a given channel. See the PLA archive on
    beam window functions
    (https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Beam_Window_Functions)
    for details on the W matrix.

    :returns: `Wl`, A 3-D ndarray of dimension (lmax + 1, 6, 6), such that the
              observed `C_\ell^{X, Y}` is `np.dot(Wl[ell], C_ell_sky)`. C_ell_sky
              should in the order (TT, EE, BB, TE, EB, TB), the same as the
              output of `hp.anafast`.
    '''
    maptypes = ('', 'hm1', 'hm2', 'even', 'odd')
    if maptype not in maptypes:
        raise ValueError('map type {} is invalid, should be one of: {}'.format(maptype, maptypes))

    if masktype not in ('fullsky', 'plikmask'):
        raise ValueError('unknown masktype {}, should be \'fullsky\' or \'plikmask\''.format(masktype))

    beamwf_file = 'Wl_R3.01_{masktype}_{freq:03}{maptype}x{freq:03}{maptype}.fits'
    beamwf_file = os.path.join(beamwf_dir, beamwf_file.format(masktype=masktype,
                                                              freq=planck_freqs[chan],
                                                              maptype=maptype))

    # Matrix of (TT, EE, BB, TE, EB, TB)^2 window function
    vectors = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    output = np.zeros((lmax + 1, 6, 6), dtype=np.double)
    # TODO get real beams for EB & TB
    output[:, 4, 4] = output[:, 5, 5] = 1

    with fitsio.FITS(beamwf_file, 'r') as fits_file:
        # There are no headers for 'EB' & 'TB' bc they are expected to have
        # negligible impact on the other data vectors
        for xi, xX in enumerate(vectors[:4]):
            this_col = fits_file[xX]
            for yi, yY in enumerate(vectors):
                xX_to_yY = this_col[xX + '_2_' + yY].read()[0]
                output[:lmax + 1, yi, xi] = xX_to_yY[:lmax + 1]

    return output
