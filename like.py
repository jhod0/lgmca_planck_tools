#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function

import healpy as hp
import numpy as np
import os
from .planck.constants import (planck_freqs, ffp8_nu4_central_freqs, ffp8_nu6_central_freqs)
from . import make_binmatrix


_ffp8_1_chan_fmt = os.path.join(os.path.dirname(__file__),
                                'data', 'ffp8.1_cls', 'R3_beams',
                                'ffp8.1_cmb_scl_{freq:03}_{real:04}.fits')


def gen_ffp8_1_like(planck_channel, ffp8_1_realization, cov_fname,
                    lmin=70, lmax=2000, delta_ell=30):
    '''
    Create a Cobaya likelihood function for a single channel of an FFP8.1
    realization.

    planck_channel: int, one of [30, 44, 70, ..., 857]
    ffp8_1_realization: int, [0..99], realization number
    cov_fname: File with the data covariance matrix
    '''
    freqi = np.arange(9)[planck_channel == planck_freqs][0]
    bmat = make_binmatrix(lmin=lmin, lmax=lmax, dl=delta_ell)

    # Bin cov & its inverse
    cov = np.loadtxt(cov_fname)
    binned_cov = np.dot(bmat, np.dot(cov[:lmax + 1, :lmax + 1], bmat.T))
    inv_binned_cov = np.linalg.inv(binned_cov)
    cov_det = np.linalg.det(binned_cov)

    # Normalization of likelihood
    k = binned_cov.shape[0]
    loglike_norm = -0.5 * (k*np.log(2*np.pi) + np.log(cov_det))

    # Load rayleigh templates
    rayleigh_base_dir = os.path.join(os.path.dirname(__file__),
                                     'data', 'rayleigh', 'ffp8_1_rayleigh', 'Rayleigh')
    header = np.zeros((2, 5))
    header[1, 0] = 1
    nu_ref = 857
    rayleigh_base = np.vstack((header,
                               np.loadtxt(os.path.join(rayleigh_base_dir,
                                          'planck_FFP8_1_total_lensedCls.dat'))))
    rayleigh_nu4 = np.vstack((header,
                              np.loadtxt(os.path.join(rayleigh_base_dir,
                                         'planck_FFP8_1_nu4_lensedCls.dat_6_6'))))
    rayleigh_nu6 = np.vstack((header,
                              np.loadtxt(os.path.join(rayleigh_base_dir,
                                         'planck_FFP8_1_nu6_lensedCls.dat_6_6'))))
    rayleigh_nu8 = np.vstack((header,
                              np.loadtxt(os.path.join(rayleigh_base_dir,
                                         'planck_FFP8_1_nu8_lensedCls.dat_6_6'))))

    # Compute effective rayleigh contribution to TT here
    nu4_eff = ffp8_nu4_central_freqs[freqi]
    nu6_eff = ffp8_nu6_central_freqs[freqi]
    rayleigh_contrib = 0
    for template, nu_eff, pow in [(rayleigh_nu4, nu4_eff, 4),
                                  (rayleigh_nu6, nu6_eff, 6),
                                  (rayleigh_nu8, nu6_eff, 8)]:
        rayleigh_contrib += (template[:, 1] - rayleigh_base[:, 1]) * (nu_eff / nu_ref)**pow

    # Data vector
    Dl_input = hp.read_cl(_ffp8_1_chan_fmt.format(freq=planck_channel,
                                                  real=ffp8_1_realization))
    Dl_input = Dl_input[:lmax + 1] - rayleigh_contrib[:lmax + 1]
    binned_Dl_input = np.dot(bmat, Dl_input[:lmax + 1])

    def my_like(# Declaration of our theory requirements
                _theory={'Cl': {'tt': lmax}},
                # Declaration of available derived parameters
                _derived={}):
        # Dl from theory (the `ell_factor` argument makes Dl)
        Dl_theory = _theory.get_Cl(ell_factor=True)['tt'][:lmax+1]
        binned_Dl_theory = np.dot(bmat, Dl_theory)

        diff = binned_Dl_theory - binned_Dl_input
        chisq = np.dot(diff, np.dot(inv_binned_cov, diff))

        return loglike_norm - chisq/2

    return my_like
