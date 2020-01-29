#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import numpy as np


def make_binmatrix(lmin, lmax, dl, mean=False):
    nbin, leftover = divmod(lmax - lmin + 1, dl)
    if leftover > 0:
        nbin += 1

    binmatrix = np.zeros((nbin, lmax + 1), dtype=int)
    for bini, starti in enumerate(range(lmin, lmax + 1, dl)):
        binmatrix[bini, starti:starti+dl] = 1

    if mean:
        bin_sizes = np.dot(binmatrix, np.ones(lmax + 1))
        return np.dot(np.diag(1 / bin_sizes), binmatrix)
    return binmatrix
