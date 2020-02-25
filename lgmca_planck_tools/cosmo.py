#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import camb
# from camb import model, initialpower
import numpy as np
import os

# FFP8 cosmology, flat \Lambda-CDM
fiducial_ffp8_params = {
    'ombh2': 0.0222, 'omch2': 0.1203,
    'H0': 67.12, 'tau': 0.065,
    'As': 2.09e-9, 'ns': 0.96,
    'omnuh2': 0.00064}

# FFP8.1 cosmology, slight adjustment to 2015 best-fit
fiducial_ffp8_1_params = {
    'ombh2': 0.0223, 'omch2': 0.1184,
    'H0': 67.87, 'tau': 0.067,
    'As': 2.14e-9, 'ns': 0.97,
    'omnuh2': 0.00065}


def compute_ffp8(ombh2, omch2, omnuh2, H0, tau, As, ns):
    r'''
    Computes a lensed TT `D_\ell` power spectrum given a set of cosmological
    parameters.
    '''
    Neff = 3.046
    params = camb.CAMBparams(omnuh2=omnuh2, H0=H0, ombh2=ombh2, omch2=omch2,
                             num_nu_massive=1, num_nu_massless=Neff - 1,
                             nu_mass_degeneracies=[Neff/3],
                             nu_mass_fractions=[1.0],
                             nu_mass_numbers=[1],
                             tau=tau, WantTransfer=True,
                             # By default uses PArthENoPE
                             bbn_predictor=camb.bbn.BBN_table_interpolator())

    params.InitPower.set_params(As=As, ns=ns)#, r=0)
    params.Transfer.accurate_massive_neutrinos = True
    params.Transfer.high_precision = True
    params.set_for_lmax(2500, lens_potential_accuracy=1)

    params.Reion.use_optical_depth = True
    params.Reion.optical_depth = tau

    result = camb.get_results(params)
    return result.get_cmb_power_spectra(params, CMB_unit='muK')


class RayleighTemplate:
    '''
    A template of the effect of Rayleigh scattering on CMB temperature and
    polarization anisotropies, derived from a fixed cosmology.

    By default, it loads the templates for the fiducial FFP8.1 cosmology
    derived from the `rayleigh` branch of CAMB.
    '''
    __rayleigh_data_path = os.path.join(os.path.dirname(__file__),
                                        'data', 'rayleigh', 'ffp8_1_rayleigh',
                                        'Rayleigh')
    def __init__(self,
                 path_nu4=os.path.join(__rayleigh_data_path,
                                       'planck_FFP8_1_nu4_lensedCls.dat'),
                 path_nu6=os.path.join(__rayleigh_data_path,
                                       'planck_FFP8_1_nu6_lensedCls.dat'),
                 path_nu8=os.path.join(__rayleigh_data_path,
                                       'planck_FFP8_1_nu8_lensedCls.dat')):
        self.nu_ref = np.loadtxt(path_nu4 + '_freqs')[5]

        # Load rayleigh templates
        header = np.zeros((2, 5))
        header[1, 0] = 1
        self.rayleigh_base = np.vstack((header, np.loadtxt(path_nu4)))
        self.rayleigh_nu4 = np.vstack((header, np.loadtxt(path_nu4 + '_6_6')))
        self.rayleigh_nu6 = np.vstack((header, np.loadtxt(path_nu6 + '_6_6')))
        self.rayleigh_nu8 = np.vstack((header, np.loadtxt(path_nu8 + '_6_6')))

    def rayleigh_contrib(self, nu4_eff, nu6_eff=None, nu8_eff=None, lmax=2000):
        '''
        Returns the total expected rayleigh contribution to each
        D_\\ell = (\\ell (\ell + 1) / 2pi) C_\\ell spectra, at a given set of
        effective frequencies. Effective frequencies of different orders may
        differ from each other due to a channel's bandpass.

        returns: A np.array of doubles of shape (lmax + 1, 5). Each row
                 corresponds to a multipole moment \\ell, and each column
                 corresponds to (\\ell, TT, EE, BB, TE)
        '''
        if nu6_eff is None:
            nu6_eff = nu4_eff
        if nu8_eff is None:
            nu8_eff = nu6_eff

        output = 0
        for template, nu_eff, pow in [(self.rayleigh_nu4, nu4_eff, 4),
                                      (self.rayleigh_nu6, nu6_eff, 6),
                                      (self.rayleigh_nu8, nu6_eff, 8)]:
            output += (template - self.rayleigh_base)[:lmax + 1, :] * (nu_eff / self.nu_ref) ** pow
        output[:, 0] = np.arange(lmax + 1)
        return output

    def TT(self, nu4_eff, nu6_eff=None, nu8_eff=None, lmax=2000):
        '''
        Returns the expected rayleigh contribution to the temperature
        autocorrelation spectrum D_\\ell^{TT}, in units of (mu K^2).

        Each nu*_eff has the same meaning as in `rayleigh_contrib`.

        returns: A 1D np.array of size (lmax + 1), for the rayleigh contribution
                 at each \\ell from 0 to lmax.
        '''
        return self.rayleigh_contrib(nu4_eff, nu6_eff=nu6_eff,
                                     nu8_eff=nu8_eff, lmax=lmax)[:, 1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # omnuh2s = np.linspace(0.0, 0.001, 12)
    omnuh2s = np.linspace(0.0, 0.001, 11)
    pspecs = []

    plt.figure()
    plt.suptitle('Raw CAMB')
    for omnuh2 in omnuh2s:
        fiducial_ffp8_1_params['omnuh2'] = omnuh2
        res = compute_ffp8(**fiducial_ffp8_1_params)
        plt.plot(res['lensed_scalar'][:, 0],
                 label='omnuh2 = {:.02}'.format(omnuh2))
        pspecs.append(res['lensed_scalar'][:, 0])
        print()
    plt.legend()

    plt.show()
