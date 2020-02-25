from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
import healpy as hp
import numpy as np

from .. import make_binmatrix
from ..cosmo import RayleighTemplate
from ..planck.constants import (planck_freqs, ffp8_nu4_central_freqs, ffp8_nu6_central_freqs)


class FFP8Like(Likelihood):
    '''
    Simple likelihood for directly using the planck Full Focal Plane (FFP)
    simulations as a data vector for an MCMC sampling run.
    '''

    def initialize(self):
        # Load data vector and covariance
        self.data_vector = hp.read_cl(self.data_vector_file)[:self.lmax + 1]
        self.data_covariance = np.loadtxt(self.cov_file)[:self.lmax + 1, :self.lmax + 1]

        if self.do_rayleigh:
            try:
                freqi = np.arange(9)[self.freq == planck_freqs][0]
            except IndexError:
                msg = '{} is not a known planck band, try one of: {}'
                raise LoggedError(self.log,
                                  msg.format(self.freq, list(planck_freqs)))

            # Compute & subtract the expected Rayleigh contribution for this
            # specific planck channel
            rayleigh_template = RayleighTemplate()
            nu4_eff = ffp8_nu4_central_freqs[freqi]
            nu6_eff = ffp8_nu6_central_freqs[freqi]
            self.data_vector -= rayleigh_template.TT(nu4_eff=nu4_eff, nu6_eff=nu6_eff, lmax=self.lmax)

        # Load binning & bin
        self.binmat = make_binmatrix(self.lmin, self.lmax, self.dl)
        self.binned_data_vector = np.dot(self.binmat, self.data_vector)
        self.binned_data_cov = np.dot(self.binmat, np.dot(self.data_covariance, self.binmat.T))
        self.inv_binned_cov = np.linalg.inv(self.binned_data_cov)

        # Normalization of likelihood
        cov_det_sign, log_cov_det = np.linalg.slogdet(self.binned_data_cov)
        if cov_det_sign < 0:
            raise LoggedError(self.log, 'determinant of covariance is negative')

        k = self.binned_data_cov.shape[0]
        self.loglike_norm = -0.5 * (k*np.log(2*np.pi) + log_cov_det)
        if np.isinf(self.loglike_norm):
            raise LoggedError(self.log, 'normalization of likelihood is infinite')

    def add_theory(self):
        self.theory.needs(Cl={'tt': self.lmax})

    def logp(self, **param_values):
        expected_cls = self.theory.get_Cl(ell_factor=True)['tt'][:self.lmax + 1]
        binned_expected = np.dot(self.binmat, expected_cls)

        diff = binned_expected - self.binned_data_vector
        chisq = np.dot(diff, np.dot(self.inv_binned_cov, diff))
        if np.isinf(chisq) or np.isnan(chisq):
            raise LoggedError(self.log, 'Invalid chisq: {}'.format(chisq))

        return self.loglike_norm - chisq/2
