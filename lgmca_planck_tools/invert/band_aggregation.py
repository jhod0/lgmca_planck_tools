'''
Aggregates separate estimates for CMB bands
'''
from __future__ import division, print_function

import fitsio
import healpy as hp
import numpy as np

from . import log


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
        res =np.linspace(0,size,size+1)
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


def band_aggregation(tabfname, Tabbeam, out_fwhm=5, lmax=4000,
                     cutoffs=[20, 100, 350, 600, 1200],
                     dcuts=[15, 50, 75, 150, 300],
                     outnside=2048,
                     logger=log.null_logger):
    '''
    Aggregates each CMB estimation band into a single map.

    The bands are assumed to be in increasing order by resolution, with the
    beam of band i, B_{i, \ell}, defined by `Tabbeam[i, \ell]`.
    '''

    alms_CMB = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    ls_CMB, ms_CMB = hp.Alm.getlm(lmax)

    # Beam stuff
    # agg_filts = which estimation bands to use for which \ells in the output
    # eff_beam = "effective beam", the band beams weighted by agg_filts
    # target_beam = the beam to use in the output: gaussian with fwhm = out_fwhm
    agg_filts = GetAggFilters(Tabbeam, cutoffs, dcuts)
    eff_beam = (agg_filts * Tabbeam).sum(axis=0)
    target_beam = getbeam(fwhm=out_fwhm, lmax=lmax)
    band_alms = []

    # Iterate over each band to perform the aggregation
    # TODO: possibly `multiprocessing` parallelize this?
    nbands = agg_filts.shape[0]
    for nband in range(nbands):
        with log.Timer(logger, 'Aggregating band {}', nband):
            fname = tabfname.format(nband=nband)
            # Load the map and convert nested -> ring order
            # NB the nested -> ring conversion!!!
            tcmbmap = hp.reorder(fitsio.FITS(fname)[0].read(), n2r=True)
            logger.debug('map loaded and converted to ring format from: {}'.format(fname))

            # Convert this map to Alms
            these_alms = hp.map2alm(tcmbmap)
            this_lmax = hp.Alm.getlmax(these_alms.size)
            ls_band, ms_band = hp.Alm.getlm(this_lmax)

            # Record these_alms
            band_alms.append(these_alms.copy())

            # Weight these ones appropriately
            hp.almxfl(these_alms, agg_filts[nband], inplace=True)

            # Aggregate these Alms
            cmb_msk = (ls_CMB <= this_lmax) & (ls_CMB <= lmax)
            alms_CMB[cmb_msk] += these_alms[(ls_band <= this_lmax) & (ls_band <= lmax)]

    # The 'ideal beam' just softens the drop to zero at lmax
    min_lmin = min(3200, lmax-300)
    ideal = getidealbeam(lmax + 1, lmin=min_lmin, lmax=lmax)

    # Convert to map
    hp.almxfl(alms_CMB, target_beam * ideal**2 / eff_beam[:lmax+1], inplace=True)
    map = hp.alm2map(alms_CMB, nside=outnside, verbose=False)
    # Convert K -> \mu K
    cls = hp.alm2cl(1e6 * alms_CMB)
    ells = np.arange(cls.size)
    ll1 = ells * (ells + 1) / (2 * np.pi)

    dls = ll1 * cls / (hp.pixwin(2048, lmax=ells.size-1) * target_beam[:ells.size])**2

    return band_alms, target_beam, dls, map
