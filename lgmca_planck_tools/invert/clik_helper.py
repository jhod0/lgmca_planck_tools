'''
Code for helping generate Clik directories
'''

from __future__ import division, print_function
from astropy.io import fits
import fitsio
import healpy as hp
import numpy as np
import os


def write_to_fits(fname, data):
    data = fits.PrimaryHDU(data=np.copy(data))
    data.writeto(fname)


class Clik(object):
    r'''
    A helper class for writing `.clik` "files" (they're actually directories)
    for the modified LGMCA version of CAMspec.

    params:
        data: The `D_\ell = \ell (\ell + 1) C_\ell / (2 \pi)` CMB spectrum, in
              `\mu K^2`. Should be a 1D array representing [`D_0`, `D_1`, ..., `D_{\ell_{max}}`].
        cov: Covariance matrix of `D_\ell`s, dimension `\ell_{max}` x `\ell_{max}`. Units `\mu K^4`.
        lrange: A 2-tuple of (`\ell_{min}`, `\ell_{max}`).
        dl: `\Delta \ell`
    '''
    def __init__(self, data, cov, lrange, dl=30):
        self.lmin, self.lmax = lrange
        nll = self.lmax - self.lmin + 1
        # FIXME - there is an error which causes the binned covariance matrix
        # to be singular. Must be an OBOE or rounding error in the line below
        nbin = nll // dl

        # Binmatrix_fk is such that np.dot(binmatrix_fk, Fk) = binned Fk,
        # where Fk is a 1d array of shape (nll,)
        self.binmatrix_fk = np.zeros((nbin, nll), dtype=int)
        for i in range(nbin):
            self.binmatrix_fk[i, i*dl:(i+1)*dl] = 1

        # Bin our data
        self.X = np.dot(self.binmatrix_fk, data[self.lmin:self.lmax+1])

        # CAMspec works in C_\ell / (2 \pi), so to convert to (binned D_\ell),
        # we must first convert to D_\ell by multiplying by \ell * (\ell + 1),
        # then binning
        ll1 = np.arange(self.lmin, self.lmax + 1, dtype=np.double) * np.arange(self.lmin + 1, self.lmax + 2, dtype=np.double)
        cl_2pi_to_dl = np.diag(ll1)
        self.cl_2pi_to_binned_dl = np.dot(self.binmatrix_fk, cl_2pi_to_dl)

        self.covmat_binned = np.dot(self.binmatrix_fk, np.dot(cov[self.lmin:self.lmax+1, self.lmin:self.lmax+1], self.binmatrix_fk.T))
        self.inv_covmat_binned = np.linalg.inv(self.covmat_binned)

        # Foreground templaets copied from jupyter notebook
        self.lgmca_cib = np.ones(self.lmax + 1) * (2 * np.pi) / (3000 * 3001)
        LL = np.arange(self.lmax + 1)
        Lfactor = LL * (LL + 1) / (2.0 * np.pi)
        Lfactor[0] = 1
        self.lgmca_ps = LL**0.8 * (1.0 /(3000.0**0.8)) / Lfactor 

    @staticmethod
    def ffp8_likelihood(inversion_dir, lrange, cl_file='FFP8_v1_aggregated_cls.fits', cov='cosmicvar'):
        lmin, lmax = lrange
        beam = np.loadtxt(os.path.join(inversion_dir, 'FFP8_v1_aggregated_beam.txt'))

        ll1 = np.arange(lmax + 1) * np.arange(1, lmax + 2) / (2 * np.pi)

        # Convert K^2 to \mu K^2 and adjust for beam
        cmb_cl = 1e12 * hp.read_cl(os.path.join(inversion_dir, cl_file)) / beam**2
        cmb_dl = ll1 * cmb_cl[:ll1.shape[0]]

        if cov == 'cosmicvar':
            cosmicvar = (2 / (2 * np.arange(lmax + 1) + 1)) * cmb_dl**2
            cov = np.diag(cosmicvar)
        elif isinstance(cov, str):
            # Assume it is path to covariance matrix
            cov = np.loadtxt(cov)
        else:
            # Assume it IS covariance matrix
            cov = np.array(cov, dtype='f8')

        return Clik(cmb_dl, cov, lrange)

    @property
    def nll(self):
        return self.binmatrix_fk.shape[1]

    @property
    def nbin(self):
        return self.binmatrix_fk.shape[0]

    def write(self, path):
        '''
        Creates a `.clik` directory at `path`.

        `path` should be a string representing a valid file path (ideally ending in `.clik/`)
        which does not yet exist.
        '''
        if os.path.exists(path):
            msg = 'Path {} already exists: refusing to overwrite'.format(path)
            raise RuntimeError(msg)

        os.mkdir(path)
        os.mkdir(os.path.join(path, 'clik'))
        os.mkdir(os.path.join(path, 'clik', 'lkl_0'))

        # Briefly, a .clik dir looks like:
        # example.clik/
        #   |-- mdb                         = empty file
        #   |-- clik/
        #         |-- _mdb                  = two params: n_lkl_object, check_value
        #         |-- lmax                  = shape=(6,) fits file: TT, EE, BB, TE, TB, EB (FIXME: is this right?)
        #         |-- lkl_0/
        #               |-- _mdb            = lots of parameters
        #               |-- beam_cov_inv    = FITS shape=(400,) of 0.0's
        #               |-- beam_modes      = don't know what this is yet, a FITS shape=(60020,) file of all 0.0's
        #               |-- c_inv           = inverse of binned covariance matrix
        #               |-- has_cl          = FITS shape=(6,) array of [1, 0, 0, 0, 0, 0]
        #               |-- ksz             = FITS shape=(5001,)
        #               |-- lgmca_cib       = FITS shape=(2001,), CIB power template
        #               |-- lgmca_ps        = FITS shape=(2001,), point src power template, constant
        #               |-- lmaxX           = FITS shape=(1,) of [2000]
        #               |-- lminX           = FITS shape=(1,) of [30]
        #               |-- np              = FITS shape=(1,) of [1971]
        #               |-- npt             = FITS shape=(1,) of [1]
        #               |-- nrebin          = FITS shape=(1,) of [65]
        #               |-- rebinM          = FITS shape=(nbin*nll,) of rebinning_matrix.flatten()
        #               |-- tsz             = FITS shape=(5001,)
        #               |-- tszXcib         = FITS shape=(5001,)
        #               |-- X               = finally, the D_\ell data vector: FITS 1d vector of size (nrebin,)

        with open(os.path.join(path, '_mdb'), 'w'):
            # We just want to create an empty file
            pass

        with open(os.path.join(path, 'clik/_mdb'), 'w') as f:
            f.writelines('\n'.join(['n_lkl_object int 1',
                                   'check_value float -3908.708434']))

        # lmaxs = fits.PrimaryHDU(data=np.array([self.lmax, -1, -1, -1, -1, -1], dtype='i8'))
        # lmaxs.writeto(os.path.join(path, 'clik/lmax'))
        write_to_fits(os.path.join(path, 'clik/lmax'), np.array([self.lmax, -1, -1, -1, -1, -1], dtype='i8'))

        with open(os.path.join(path, 'clik/lkl_0/_mdb'), 'w') as f:
            # Most of these are copied from working `.clik` with no knowledge of what they do
            lines = ['nX int {}'.format(self.nll),
                     'num_modes_per_beam int 5',
                     'lmin int {}'.format(self.lmin),
                     'has_dust int 0',
                     'cov_dim int 20',
                     'lmax_sz int 5000',
                     'lmax int {}'.format(self.lmax),
                     'beam_Nspec int 4',
                     'has_calib_prior int 1',
                     'Nspec int 1',
                     'lkl_type str LGMCAbinCAMspec',
                     # What the hell is this one?
                     'pipeid str e61cec87-3a37-43ca-8ed1-edcfcaf5c00a',
                     'unit int 1',
                     'beam_lmax int 3000',
                     'nbins int 0',
                     'nrebin int {}'.format(self.nbin)]
            f.writelines('\n'.join(lines))

        # Write 'em all
        lkl_0_path = os.path.join(path, 'clik/lkl_0')
        write_to_fits(os.path.join(lkl_0_path, 'beam_cov_inv'), np.zeros(400, dtype=np.double))
        write_to_fits(os.path.join(lkl_0_path, 'beam_modes'), np.zeros(60020, dtype=np.double))
        write_to_fits(os.path.join(lkl_0_path, 'c_inv'), self.inv_covmat_binned.flatten())
        write_to_fits(os.path.join(lkl_0_path, 'has_cl'), np.array([1, 0, 0, 0, 0, 0]))
        write_to_fits(os.path.join(lkl_0_path, 'ksz'), np.zeros(5001, dtype=np.double))
        write_to_fits(os.path.join(lkl_0_path, 'lgmca_cib'), self.lgmca_cib / (2 * np.pi))
        write_to_fits(os.path.join(lkl_0_path, 'lgmca_ps'), self.lgmca_ps / (2 * np.pi))
        write_to_fits(os.path.join(lkl_0_path, 'lmaxX'), np.array([self.lmax]))
        write_to_fits(os.path.join(lkl_0_path, 'lminX'), np.array([self.lmin]))
        write_to_fits(os.path.join(lkl_0_path, 'np'), np.array([self.nll]))
        write_to_fits(os.path.join(lkl_0_path, 'npt'), np.array([1]))
        write_to_fits(os.path.join(lkl_0_path, 'nrebin'), np.array([self.nbin]))
        write_to_fits(os.path.join(lkl_0_path, 'rebinM'), self.cl_2pi_to_binned_dl.flatten())
        write_to_fits(os.path.join(lkl_0_path, 'tsz'), np.zeros(5001, dtype=np.double))
        write_to_fits(os.path.join(lkl_0_path, 'tszXcib'), np.zeros(5001, dtype=np.double))
        write_to_fits(os.path.join(lkl_0_path, 'X'), self.X)
