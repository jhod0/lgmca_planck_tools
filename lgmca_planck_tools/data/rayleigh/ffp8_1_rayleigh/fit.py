#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import numpy as np


def make_R_matrices(R4, R6, nus, nu0=857):
    eye = np.eye(2)
    return [eye + R4 * (nu/nu0)**4 + R6 * (nu/nu0)**6
            for nu in nus]


def do_rayleigh(Cl, R4, R6, nus, nu0=857):
    # Cl should be 2x2 matrix of:
    # [[C^TT, C^TE],
    #  [C^ET, C^EE]]
    Rs = make_R_matrices(R4, R6, nus, nu0=nu0)
    return np.array([np.dot(R, np.dot(Cl, R.T))
                     for R in Rs])


def residual(args, Cls, nu_Cls, nus, ell, nu0):
    assert Cls.shape[-1] == 5
    assert nu_Cls.shape[0] == len(nus)
    assert Cls.shape[0] == nu_Cls.shape[1]

    ell_msk = Cls[:, 0] == ell

    input_Cls = np.zeros((2, 2))
    row = Cls[ell_msk][0]
    # Row is now (ell, TT, EE, BB, TE/ET)
    input_Cls[0, 0] = row[1]
    input_Cls[0, 1] = input_Cls[1, 0] = row[-1]
    input_Cls[1, 1] = row[2]
    # print(input_Cls)

    R4, R6 = args.reshape((2, 2, 2))
    predicted = do_rayleigh(input_Cls, R4, R6, nus=nus, nu0=nu0)
    cosmicvar = input_Cls**2 * 2 / (2 * ell + 1)
    cosmicvar = 1

    # We don't want to weight mostly by the TEs, EEs, so we arbitrarily
    # increase their error bars by sqrt(10)
    # cosmicvar[0, 1] *= 10
    # cosmicvar[1, 0] *= 10
    # cosmicvar[1, 1] *= 10

    residual = []
    # print(predicted.shape)
    # print(nu_Cls.shape)
    for nu_Cl, pred_Cl in zip(nu_Cls[:, ell_msk][:, 0, :], predicted):
        # print(nu_Cl)
        nu_Cl = np.array([[nu_Cl[1], nu_Cl[-1]],
                          [nu_Cl[-1], nu_Cl[2]]])
        residual.append(((nu_Cl - pred_Cl) / np.sqrt(cosmicvar)).flatten())

    return 100*np.concatenate(residual)


if __name__ == '__main__':
    base_Cls = np.loadtxt('Rayleigh/planck_FFP8_1_total_lensedCls.dat')
    nu_freqs = np.loadtxt('Rayleigh/planck_FFP8_1_total_lensedCls.dat_freqs')

    nu_Cls = []
    for i in range(2, 7):
        nu_Cls.append(np.loadtxt('Rayleigh/planck_FFP8_1_total_lensedCls.dat_{0}_{0}'.format(i)))
    nu_Cls = np.array(nu_Cls)
