#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function

import healpy as hp
import numpy as np
import os
from ..planck.constants import (planck_freqs, ffp8_nu4_central_freqs, ffp8_nu6_central_freqs)
from .. import make_binmatrix


def getbeam(fwhm=5, lmax=512):
    '''
    The beam used in LGMCA output
    '''

    tor = 0.0174533
    F = fwhm / 60. * tor
    l = np.linspace(0,lmax,lmax+1)
    ell = l*(l+1)
    bl = np.exp(-ell*F*F /16./np.log(2.))

    return bl


_ffp8_1_chan_fmt = os.path.join(os.path.dirname(__file__),
                                'data', 'ffp8.1_cls', 'R3_beams',
                                'ffp8.1_cmb_scl_{freq:03}_{real:04}.fits')

def bare_gen_like(Dl_input, cov,
                  lmin, lmax, delta_ell):
    bmat = make_binmatrix(lmin, lmax, dl=delta_ell)

    # Bin cov & its inverse
    binned_cov = np.dot(bmat, np.dot(cov[:lmax + 1, :lmax + 1], bmat.T))
    inv_binned_cov = np.linalg.inv(binned_cov)
    cov_det_sign, log_cov_det = np.linalg.slogdet(binned_cov)
    if cov_det_sign < 0:
        raise ValueError('determinant of covariance is negative')

    # Normalization of likelihood
    k = binned_cov.shape[0]
    loglike_norm = -0.5 * (k*np.log(2*np.pi) + log_cov_det)
    if np.isinf(loglike_norm):
        raise ValueError('normalization of likelihood is infinite')

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
        if np.isinf(chisq):
            raise ValueError('chisq is infinite')

        return loglike_norm - chisq/2

    return my_like


def gen_lgmca_like(lgmca_file, cov_file,
                   lmin=70, lmax=2000, delta_ell=30):
    # Load covariance
    cov = np.loadtxt(cov_file)

    # Add l (l + 1) factor (i.e. convert cl -> dl)
    ells = np.arange(lmax + 1)
    ll1 = ells * (ells + 1) / (2 * np.pi)

    # Load data vector
    dls = hp.read_cl(lgmca_file)[:lmax + 1] / (getbeam(5, lmax) * hp.pixwin(2048, lmax=lmax))**2

    # 1e12 is K^2 -> \mu K^2
    return bare_gen_like(1e12 * ll1 * dls, cov, lmin=lmin, lmax=lmax, delta_ell=delta_ell)


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

    # Load covariance
    cov = np.loadtxt(cov_fname)

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

    return bare_gen_like(Dl_input, cov, lmin=lmin, lmax=lmax, delta_ell=delta_ell)
