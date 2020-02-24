#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import healpy as hp
import numpy as np

from .beam import r3_channel_beams
from .constants import (ffp8_nu4_central_freqs, ffp8_nu6_central_freqs)


def make_big_R(R4, R6, base_nu4, base_nu6,
               include_T_only=True):
    if include_T_only:
        nxlms = 16
    else:
        nxlms = 14
    output = np.zeros((nxlms, 2), dtype=np.complex)

    for i in range(9):
        n4 = ffp8_nu4_central_freqs[i] / base_nu4
        n6 = ffp8_nu6_central_freqs[i] / base_nu6
        thisR = np.eye(2) + R4*n4**4 + R6*n6**6
        if i < 7:
            output[2*i:2*(i+1), :] += thisR
        elif include_T_only:
            output[2*7 + (i - 7), :] += thisR[0, :]
        else:
            break

    return output


def rayleigh_residual(data, beams, R4, R6, base_nu4, base_nu6, Xs,
                      normalization=1):
    outputs = []
    R = make_big_R(R4, R6, base_nu4, base_nu6)

    # print('data shape:', data.shape)
    # print('xs shape:', Xs.shape)
    for m, (datum, X) in enumerate(zip(data, Xs)):
        beamed_rayleighed = beams * np.dot(R, X)
        diff = (datum.flatten() - beamed_rayleighed) / normalization
        # print('diff shape, m = {}:'.format(m), diff.shape)
        outputs.append(diff.real)

        # For m == 0, the imaginary component should be zero
        if m > 0:
            outputs.append(diff.imag)

    return np.concatenate(outputs)


def pack_args(beams, r4, r6, xs, nu_ref, beam_ref, ell):
    # Skip `beam_ref`
    beams = np.concatenate((beams[:2*beam_ref], beams[2*(beam_ref+1):]))
    xs = np.dstack((xs.real, xs.imag))
    return np.concatenate((beams.flatten(), r4.flatten(), r6.flatten(), xs.flatten()))


def unpack_args(args, nu_ref, beam_ref, ell, reference_beams=r3_channel_beams):
    nbeams = 14
    beams, args = args[:nbeams], args[nbeams:]
    new_beams = np.zeros(16)

    for i in range(7):
        if i == beam_ref:
            new_beams[2*i:2*(i + 1)] = reference_beams[beam_ref, ell]
            continue
        new_beams[2*i:2*(i+1)], beams = beams[:2], beams[2:]
    assert beams.size == 2
    new_beams[-2:] = beams

    r4, args = args[:4].reshape((2, 2)), args[4:]
    r6, args = args[:4].reshape((2, 2)), args[4:]

    if (args.size % 4) != 0:
        raise ValueError('Invalid argument - not sure how to parse')
    xs = args.reshape((args.size // 4, 2, 2))
    xs = xs[:, :, 0] + 1j * xs[:, :, 1]

    return new_beams, r4, r6, xs


def make_residual_function(alms, nu_ref, beam_ref, ell, reference_beams=r3_channel_beams):
    # Alms should be (9 channels, 3 fields (TEB), hp.Alm.getsize(lmax))
    assert len(alms.shape) == 3
    assert alms.shape[0] == 9
    assert alms.shape[1] == 3

    nside = hp.Alm.getlmax(alms.shape[-1])
    ells, ems = hp.Alm.getlm(nside)

    all_Ts_data = alms[:, 0, ells == ell]
    all_Es_data = alms[:, 1, ells == ell]
    normalization_T = np.sqrt((all_Ts_data.conj() * all_Ts_data).real.sum() / (2 * ell + 1))
    normalization_E = np.sqrt((all_Es_data.conj() * all_Es_data).real.sum() / (2 * ell + 1))

    # Provide a normalization for each T & E
    normalization = np.zeros((8, 2))
    normalization[:, 0] = normalization_T
    normalization[-1, :] = normalization_T
    normalization[:-1, 1] = normalization_E
    normalization[normalization == 0] = 1
    # big_normalization = np.concatenate([normalization.flatten()]*(ell + 1))
    # print('big norm:', big_normalization.shape)

    all_data = np.zeros((ell + 1, 8, 2), dtype=np.complex)
    for m in range(ell + 1):
        # First seven channels - T & E
        all_data[m, :-1, :] = alms[:7, :2, (ells == ell) & (ems == m)][:, :, 0]
        # Last channel - just T
        all_data[m, -1, :] = alms[7:9, 0, (ells == ell) & (ems == m)][:, 0]

    base_nu4 = ffp8_nu4_central_freqs[nu_ref]
    base_nu6 = ffp8_nu6_central_freqs[nu_ref]
    def residual(args):
        beams, r4, r6, Xs = unpack_args(args, nu_ref, beam_ref, ell)
        res = rayleigh_residual(all_data.reshape((ell + 1, -1)),
                                beams, r4, r6,
                                base_nu4, base_nu6, Xs,
                                normalization=normalization.flatten())
        # print('residual shape:', res.shape)
        return res

    default_beams = []
    for i in range(7):
        default_beams += [reference_beams[i, ell]]*2
    default_beams.extend(reference_beams[-2:, ell])

    r4 = np.zeros((2, 2))
    r6 = np.zeros((2, 2))

    Xs = []
    for m in range(ell + 1):
        X = alms[beam_ref, :2, (ells == ell) & (ems == m)][0]
        Xs.append(X / reference_beams[beam_ref, ell])

    return residual, pack_args(np.array(default_beams), r4, r6, np.array(Xs), nu_ref, beam_ref, ell)
