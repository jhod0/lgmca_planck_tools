#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import camb
# from camb import model, initialpower
import numpy as np

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
    '''
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
