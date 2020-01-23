#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function

import fitsio
import healpy as hp
import numpy as np

from . import (cfact_545_857, ffp8_channel_beams, ffp8_dipole_magnitude,
               ffp8_doppler_adjs, ffp8_theta_dipole, ffp8_phi_dipole,
               planck_freqs)


def doppler_adjust(cmap, chani):
    nside = hp.npix2nside(cmap.size)
    pix = np.arange(cmap.size)
    theta, phi = hp.pix2ang(nside, pix)

    cos_great_arc = (np.cos(theta) * np.cos(ffp8_theta_dipole) +
                     np.sin(theta) * np.sin(ffp8_theta_dipole) *
                     np.cos(phi - ffp8_phi_dipole))

    factor = ffp8_dipole_magnitude * cos_great_arc
    return cmap / (1 + ffp8_doppler_adjs[chani] * factor)


def load_chan_realization(chan_fmt, chani, reali, lmax, doppler=True):
    '''
    Loads a single array of `D_\ells` from an FFP8 fits file, adjusting for the
    observation beam and (possibly) for doppler modulation. Returns the TT
    power spectrum for `\ell = 0` to `\ell = lmax`, inclusive.

    chan_fmt: Path to the .fits file.
    chani: Planck observation channel, 0 <= chani < 9. I.e. 0 means 30 GHz.
    reali: The realization number.
    doppler: Whether to remove doppler modulation from the realization.

    returns: An array of `D_\ell = \ell (\ell + 1) / (2 \pi) C_\ell`.
             A `numpy.array` of `dtype == np.double` and size `lmax + 1`.
    '''
    m = hp.read_map(chan_fmt.format(freq=planck_freqs[chani], real=reali))
    if chani >= 7:
        m /= cfact_545_857[chani - 7]

    if doppler:
        m = doppler_adjust(m, chani)

    cls = hp.anafast(1e6*m, lmax=lmax)
    ells = np.arange(lmax + 1)
    ll1 = ells * (ells + 1) / (2 * np.pi)

    nside = hp.npix2nside(m.size)
    beam = ffp8_channel_beams[chani, :lmax + 1] * hp.pixwin(nside)[:lmax + 1]

    adj_dls = ll1 * cls / beam**2
    return adj_dls


def load_realization(chan_fmt, reali, lmax=3000, doppler=True):
    '''
    Load a FFP8 realization, i.e. the `D_\ell` power spectrum for each planck
    observation channel, correcting for the channel beam and (optionally)
    doppler modulation.  Returns the TT power spectrum for `\ell = 0` to
    `\ell = lmax`, inclusive.

    chan_fmt: Path to the .fits file.
    reali: The realization number.
    doppler: Whether to remove doppler modulation from the realization.

    returns: An `np.array` of shape `(9, lmax + 1)`.
    '''
    result = np.zeros((9, lmax + 1))
    for i in range(9):
        result[i] = load_chan_realization(chan_fmt, i, reali, lmax, doppler=doppler)
    result[np.isinf(result)] = 0.0
    return result
