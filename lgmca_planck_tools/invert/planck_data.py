'''
Some useful Planck data-related numbers.

planck_freqs = the frequencies of the 9 Planck bands in GHz, low to high
cfact_545_857 = mJy/sr -> Kelvin conversion factor for two highest-freq bands
col_cmb = T_{cmb} -> T_{antenna} conversion factor
'''
from __future__ import division, print_function

import fitsio
import healpy as hp
import logging
import multiprocessing as mp
import numpy as np

from . import log

from lgmca_planck_tools.planck import r3_channel_beams
from lgmca_planck_tools.planck.constants import (
        planck_freqs,
        cfact_545_857,
        col_cmb
    )

# ==================== Data Files ============================= #

# # Tabbeam is 15 x 4001: 15 spectra of \ell \in [0, 4000]
# # First 9 rows are for the channel beams used in FFP8
# # Last 6 rows are the "common beams" (?)
# tabbeam = fitsio.FITS(tabbeam_file)[0].read()
tabbeam_file = '/home/humphrey/Saclay/planck_sim_tools/lgmca_planck_tools/data/PR2_LGMCA/restored_tabbeam_FFP8.fits'
with fitsio.FITS(tabbeam_file) as f:
    tabbeam = f[0].read()

# ==================== Individual Chan Processing ============= #

# Pixel Window Function to properly change a NSIDE=1024 map to a NSIDE=2048 map
resolution_change_pwf = hp.pixwin(2048)[:2049] / hp.pixwin(1024)[:2049]

def load_channel(chan_i, fname_fmt, logger=log.null_logger, TabbeamChan=r3_channel_beams, verbose=False):
    '''
    Loads a map for a given channel. Adjusts it to the expected beam and applies a low-pass filter.
    '''
    logger.debug('[chan {}] about to read map'.format(chan_i))
    with log.Timer(logger, '[chan {}] reading map', chan_i):
        fname = fname_fmt.format(planck_freqs[chan_i])
        logger.debug('[chan {}] reading map from {}'.format(chan_i, fname))
        data = hp.read_map(fname, verbose=verbose)

    # Last two channels are in mJ / sr, NOT in Kelvin - need to convert
    if chan_i >= 7:
        data /= cfact_545_857[chan_i - 7]

    map = hp.ud_grade(data, 2048, order_in='ring', order_out='ring')
    map[map < -1e6] = 0.0

    norm = TabbeamChan[chan_i, 0] / tabbeam[chan_i, 0]
    if chan_i < 3:
        lmax = 2500
        beameq = resolution_change_pwf*(tabbeam[chan_i, :2049] * norm / TabbeamChan[chan_i, :2049])
    else:
        lmax = 4500
        beameq = tabbeam[chan_i] * norm / TabbeamChan[chan_i]

    # Low pass filter
    with log.Timer(logger, '[chan {}] low pass filtering', chan_i):
        alms = hp.map2alm(map, use_weights=True, lmax=lmax)
        hp.almxfl(alms, beameq, inplace=True)
        adj_map = hp.reorder(hp.alm2map(alms, 2048, verbose=verbose), r2n=True)

    # Convert T_{CMB} to T_{antenna}
    return adj_map * col_cmb[chan_i]


def load_channels(fname_fmt, logger=log.null_logger, TabbeamChan=r3_channel_beams, verbose=False):
    '''
    Loads planck channels 1-9 in the files described by fname_fmt, and applies
    appropriate beam transformations and cuts.
    '''
    with log.Timer(logger, 'reading in all freq maps'):
        all_bands = np.array([hp.ud_grade(hp.read_map(fname_fmt.format(planck_freqs[chan_i]),
                                                      verbose=verbose),
                                          2048, order_in='ring', order_out='ring')
                              for chan_i in range(9)])

    all_bands[all_bands < -1e6] = 0.0

    all_bands[7] /= cfact_545_857[0]
    all_bands[8] /= cfact_545_857[1]

    output_maps = np.zeros_like(all_bands)

    with log.Timer(logger, 'filtering LFI bands'):
        alms = hp.map2alm(all_bands[:3], lmax=2500, use_weights=True, pol=False)
        for i in range(alms.shape[0]):
            norm = TabbeamChan[i, 0]
            beameq = resolution_change_pwf*(tabbeam[i, :2049] * norm / TabbeamChan[i, :2049])
            hp.almxfl(alms[i], beameq, inplace=True)
            output_maps[i] = col_cmb[i] * hp.reorder(hp.alm2map(alms[i], 2048, verbose=verbose), r2n=True)

    with log.Timer(logger, 'filtering HFI bands'):
        alms = hp.map2alm(all_bands[3:], lmax=4000, use_weights=True, pol=False)
        for i in range(alms.shape[0]):
            norm = TabbeamChan[3 + i, 0]
            beameq = norm * tabbeam[3 + i] / TabbeamChan[3 + i]
            hp.almxfl(alms[i], beameq, inplace=True)
            output_maps[3 + i] = col_cmb[3 + i] * hp.reorder(hp.alm2map(alms[i], 2048, verbose=verbose), r2n=True)

    return output_maps


def _packed_load_channel(arg):
    i, fname_fmt, logger = arg
    return load_channel(i, fname_fmt, logger=logging.getLogger(logger))


def create_cube(fname_fmt, logger=log.null_logger, parallel=True):
    '''
    Creates a 'CMB cube' by processing the 9 Planck bands. The CMB cube is a
    collection of all the input bands into a single file, with some adjustments
    to their bands and a low-pass filter applied. It is the primary input of the
    LGMCA inversion.
    '''
    n_bands = 9

    logger.debug('creating CMB cube from files:')
    for i in range(n_bands):
        logger.debug('\t' + fname_fmt.format(planck_freqs[i]))

    if parallel:
        # The process pools were behaving suspiciously, hence the debug info here.
        # (Specifically: Once the process pool was finished loading in these
        #   channels (step 1), it took about 4 minutes for the inversion (step 2)
        #   to start, even though there should only be trivial work between the two steps
        #   Since it looked like a bunch of child processes were staying open
        #   I have a hunch this delay had to do with them, but I could be wrong)
        # FIXME this is resolved/stops happening remove the debug() calls here
        logger.debug('creating process pool')
        pool = mp.Pool(min(n_bands, mp.cpu_count()))
        logger.debug('process pool created, mapping to load channels')
        res = np.array(pool.map(_packed_load_channel, [(i, fname_fmt, logger.name) for i in range(n_bands)]))
        logger.debug('channels loaded, closing process pool')
        pool.terminate()
        pool.join()
        logger.debug('process pool teminated')
        return res

    else:
        output = np.zeros((n_bands, hp.nside2npix(2048)))
        for i in range(n_bands):
            output[i] = load_channel(i, fname_fmt, logger=logger)
        return np.array(output)
