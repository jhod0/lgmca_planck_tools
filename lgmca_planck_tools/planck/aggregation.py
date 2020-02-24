#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com
'''
The aggregation used for LGMCA estimation bands.
'''

from __future__ import division, print_function
import numpy as np


def getbeam(fwhm=5, lmax=512):
    '''
    Get a beam at a given fwhm
    '''

    tor = 0.0174533
    F = fwhm / 60. * tor
    l = np.linspace(0,lmax,lmax+1)
    ell = l*(l+1)
    bl = np.exp(-ell*F*F /16./np.log(2.))

    return bl

def getidealbeam(N, lmin=512, lmax=1024, tozero=True):
    '''
    Get a "perfect" beam with a high multipole cut (L-P filter)
    '''

    def spline2(size,l,lc):
        res = np.linspace(0,size,size+1)
        res = 2.0 * l * res / (lc *size)
        tab = (3.0/2.0)*1.0 /12.0 * (( abs(res-2))**3 - 4.0* (abs(res-1))**3 + 6 *(abs(res))**3 - 4.0 *( abs(res+1))**3+(abs(res+2))**3)
        return tab

    bl = np.zeros((N,))
    bl[0:lmin] = 1.
    Np = lmax-lmin - 1
    #x = np.linspace(0,Np-1,Np) / (Np-1)*3.1416/2
    t = spline2( Np, 1, 1)

    if tozero == True:
        bl[lmin:lmax] = t
        bl[lmax::]=0.
    else:
        bl[lmin:lmax] = bl[lmin:lmax] + t * (1. - bl[lmin:lmax])

    return bl

# The boundaries between each estimation band, and the "blurring" between them
default_cutoffs = [20, 100, 350, 600, 1200]
default_dcuts = [15, 50, 75, 150, 300]

def GetAggFilters(tabbeam, cutoff, dcut):
    '''
    Define the aggregation filters
    '''

    NbrBands = np.shape(tabbeam)[0]
    lmax = np.shape(tabbeam)[1] - 1

    AFilters = np.zeros((NbrBands,lmax+1))
    totfilter = np.zeros((lmax+1,))

    for q in range(NbrBands-1):
        f = getidealbeam(lmax+1, lmin=cutoff[q] - dcut[q], lmax=cutoff[q] + dcut[q])
        AFilters[q,:] = f - totfilter
        totfilter = AFilters[q,:] + totfilter
    AFilters[NbrBands-1,:] = 1. - totfilter

    return AFilters
