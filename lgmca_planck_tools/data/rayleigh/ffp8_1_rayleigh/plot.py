#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


def load_deltas(basename):
    deltas = []
    base = np.loadtxt(basename)

    for i in range(1, 7):
        row = []
        for j in range(1, 7):
            row.append(np.loadtxt(basename + '_{}_{}'.format(i, j)))
        deltas.append(row)

    return base, np.array(deltas)


if __name__ == '__main__':
    file = 'Rayleigh/planck_FFP8_1_total_lensedCls.dat'
    base, base_deltas = load_deltas(file)
    freqs = np.loadtxt(file + '_freqs')

    fig, axs = plt.subplots(figsize=(16, 16), sharex=True, nrows=3, ncols=3)
    for i in range(2, 5):
        for j in range(2, 5):
            this_ax = axs[i-2, j-2]
            this_ax.plot(base[:, 0], (base_deltas[i, j, :, 1] / base[:, 1]) - 1,
                         color='red', label='TT')
            this_ax.plot(base[:, 0], (base_deltas[i, j, :, 2] / base[:, 2]) - 1,
                         color='blue', label='EE')
            this_ax.plot(base[:, 0], (base_deltas[i, j, :, 3] / base[:, 3]) - 1,
                         color='magenta', label='BB')
            this_ax.set_title('{}x{}'.format(freqs[i], freqs[j]))
            this_ax.grid()
    axs[0, 0].set_xlim((0, 2500))

    nu4s, nu4s_nus = load_deltas('Rayleigh/planck_FFP8_1_nu4_lensedCls.dat')
    nu6s, nu6s_nus = load_deltas('Rayleigh/planck_FFP8_1_nu6_lensedCls.dat')
    nu8s, nu8s_nus = load_deltas('Rayleigh/planck_FFP8_1_nu8_lensedCls.dat')

    # Confirm Rayleigh = O(nu^4) + O(nu^6) + O(nu^8)
    # fig, axs = plt.subplots(figsize=(16, 16), sharex=True, nrows=3, ncols=3)
    # for i in range(2, 5):
    #     for j in range(2, 5):
    #         this_ax = axs[i-2, j-2]
    #         sum_deltas = (nu4s_nus[i, j] - nu4s) + (nu6s_nus[i, j] - nu6s) + (nu8s_nus[i, j] - nu8s)
    #         total_deltas = base_deltas[i, j] - base
    #         diff = sum_deltas - total_deltas
    #         this_ax.plot(base[:, 0], diff[:, 1],
    #                      color='red', label='TT')
    #         this_ax.plot(base[:, 0], diff[:, 2],
    #                      color='blue', label='EE')
    #         this_ax.plot(base[:, 0], diff[:, 3],
    #                      color='magenta', label='BB')
    #         this_ax.set_title('{}x{}'.format(freqs[i], freqs[j]))
    #         this_ax.grid()
    # axs[0, 0].set_xlim((0, 2500))

    fig, axs = plt.subplots(figsize=(16, 16), sharex=True, nrows=4, ncols=4)
    for i in range(2, 6):
        for j in range(2, 6):
            this_ax = axs[i-2, j-2]
            this_ax.plot(nu4s[:, 0], nu4s_nus[i, j, :, 1] - nu4s[:, 1],
                         label='nu^4 TT')
            this_ax.plot(nu4s[:, 0], nu6s_nus[i, j, :, 1] - nu4s[:, 1],
                         label='nu^6 TT')
            this_ax.plot(nu4s[:, 0], nu8s_nus[i, j, :, 1] - nu4s[:, 1],
                         label='nu^8 TT')
            this_ax.plot(nu4s[:, 0], base_deltas[i, j, :, 1] - nu4s[:, 1],
                         label='total TT')
            this_ax.set_title('{}x{}'.format(freqs[i], freqs[j]))
            this_ax.grid()
    axs[0, 0].set_xlim((0, 2500))

    plt.show()
