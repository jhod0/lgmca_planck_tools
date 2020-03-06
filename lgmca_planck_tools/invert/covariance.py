'''
This is a script to compute the (noise + cosmic variance) covariance of the
LGMCA-inverted Planck data, using the FFP8/FFP8.1 simulations.
'''

from __future__ import division, print_function

import copy
import fitsio
import invert
import log
import logging
import multiprocessing as mp
import os
import shutil

import healpy as hp
import numpy as np


def do_inversion(config):
    '''
    Same thing as invert.run_inversion, just don't return anything so we don't
    waste memory
    '''
    try:
        invert.run_inversion(config)
    except RuntimeError:
        logger = logging.getLogger('lgmca_processing.' + config['name'])
        logger.error('RuntimeError occurred running: ' + str(config))
        logger.error('Trying again.')
        do_inversion(config)
        

def combine_cmb_noise(config, n, cmb_dir, noise_dir, coupling_matrix, mask):
    combined_dir = os.path.join(config['toplevel_dir'], 'ffp8_mc_combined_{:04}'.format(n))
    if not os.path.exists(combined_dir):
        os.mkdir(combined_dir)

    logger = log.make_logger('ffp8_combined_{}'.format(n),
                             log_file=os.path.join(combined_dir, 'log.txt'),
                             toplevel_log_file=os.path.join(config['toplevel_dir'], 'lgmca_postprocessing_combination_log.txt'))

    with log.Timer(logger, 'Combining CMB {0} and noise {0}'.format(n)).with_level(logging.INFO):
        with log.Timer(logger, 'Reading CMB & noise maps and combining them'):
            cmb_map = hp.read_map(os.path.join(cmb_dir, 'FFP8_v1_aggregated_cmb.fits'), verbose=False)
            noise_map = hp.read_map(os.path.join(noise_dir, 'FFP8_v1_aggregated_cmb.fits'), verbose=False)
            combined_map_ring = cmb_map + noise_map

        # No need to waste memory
        del cmb_map
        del noise_map

        combined_map = hp.reorder(combined_map_ring, r2n=True)
        cls = hp.anafast(combined_map_ring, lmax=config['matmask_maxl'], use_weights=True)

        hp.write_map(os.path.join(combined_dir, 'FFP8_v1_aggregated_map.fits'),
                     combined_map, nest=True, overwrite=True)
        hp.write_cl(os.path.join(combined_dir, 'FFP8_v1_aggregated_cls.fits'),
                    cls, overwrite=True)
        shutil.copyfile(os.path.join(cmb_dir, 'FFP8_v1_aggregated_beam.txt'),
                        os.path.join(combined_dir, 'FFP8_v1_aggregated_beam.txt'))

        with log.Timer(logger, 'Computing masked pspec and decoupling'):
            masked_powerspec = hp.anafast(combined_map_ring*mask, lmax=config['matmask_maxl'], use_weights=True)
            recovered_pspec = np.linalg.solve(coupling_matrix, masked_powerspec)
            hp.write_cl(os.path.join(combined_dir, 'mask_corrected_spectra.fits'),
                        recovered_pspec, overwrite=True)


def do_combinations(start=0, total=100):
    config = {'toplevel_dir': '/dsm/cosmo02/dataPlanck/jodonnell/ffp8_covariance/',
              'coefs_fname': '/dsm/cosmo02/sparseastro/fsureau/PR2_cosmoparams/data/FFP8_LGMCA_Band{cmap}_CoefMaps.fits',
              'structure_fname': '/dsm/cosmo02/sparseastro/fsureau/PR2_cosmoparams/temp/params_FFP8.fits',
              'band_output_name': 'FFP8_v1_LGMCA_Band0.fits',
              'mask_fname': '/dsm/cosmo02/sparseastro2/fsureau/Planck_processing/masks/PR1_Analysis_Mask_76p.fits',
              'mask_coupling_matrix_fname': '/dsm/cosmo02/dataPlanck/jodonnell/cosmoparams/jack_work/data/PR1_Analysis_Mask_76p_coupling_matrix.fits',
              'matmask_maxl': 4000}

    mask = hp.read_map(config['mask_fname'], verbose=False)
    coupling_matrix = fitsio.FITS(config['mask_coupling_matrix_fname'])[0].read()

    for n in range(start, start+total):
        cmb_dir = os.path.join(config['toplevel_dir'], 'ffp8_mc_cmb_{:04}'.format(n))
        noise_dir = os.path.join(config['toplevel_dir'], 'ffp8_mc_noise_{:05}'.format(n))
        combine_cmb_noise(config, n, cmb_dir, noise_dir, coupling_matrix, mask)


def generate_covariance(fname_fmt, output_fname, nvecs=100, beam_fname='FFP8_v1_aggregated_beam.txt'):
    first_vec = hp.read_cl(fname_fmt.format(0))
    maxl = first_vec.size - 1

    data_vecs = np.zeros((nvecs, maxl+1))

    ells = np.arange(maxl+1)
    ll1 = ells * (ells + 1) / (2 * np.pi)

    for i in range(0, nvecs):
        # Convert \deg K -> \mu \deg K
        fname = fname_fmt.format(i)
        beam_fname = os.path.join(os.path.dirname(fname), 'FFP8_v1_aggregated_beam.txt')
        data_vecs[i] = ll1 * 1e12*hp.read_cl(fname_fmt.format(i)) / (np.loadtxt(beam_fname)**2)

    covariance = np.cov(data_vecs.T)
    np.savetxt(output_fname, covariance)

    return covariance


if __name__ == '__main__':
    toplevel_dir = '/dsm/cosmo02/dataPlanck/jodonnell/ffp8_covariance/'

    cmb_config = {'toplevel_dir': toplevel_dir,
                  'coefs_fname': '/dsm/cosmo02/sparseastro/fsureau/PR2_cosmoparams/data/FFP8_LGMCA_Band{cmap}_CoefMaps.fits',
                  'structure_fname': '/dsm/cosmo02/sparseastro/fsureau/PR2_cosmoparams/temp/params_FFP8.fits',
                  'band_output_name': 'FFP8_v1_LGMCA_Band0.fits',
                  'mask_fname': '/dsm/cosmo02/sparseastro2/fsureau/Planck_processing/masks/PR1_Analysis_Mask_76p.fits',
                  'mask_coupling_matrix_fname': '/dsm/cosmo02/dataPlanck/jodonnell/cosmoparams/jack_work/data/PR1_Analysis_Mask_76p_coupling_matrix.fits',
                  'matmask_maxl': 4000}
    noise_config = copy.deepcopy(cmb_config)

    # There are 100 each of CMB and noise realizations
    configs = []
    for n in range(85, 100):
        cmb_config['name'] = 'ffp8_mc_cmb_{:04}'.format(n)
        noise_config['name'] = 'ffp8_mc_noise_{:05}'.format(n)

        cmb_config['component_fname'] = '/dsm/cosmo02/sparseastro/PLANCK_DATA/ffp8.1/mc_cmb/{0:03}/ffp8.1_cmb_scl_{0:03}_full_map_mc_' + '{:04}.fits'.format(n)
        noise_config['component_fname'] = '/dsm/cosmo02/sparseastro/PLANCK_DATA/ffp8/mc_noise/{0:03}/ffp8_noise_{0:03}_full_map_mc_' + '{:05}.fits'.format(n)

        configs.append(copy.deepcopy(cmb_config))
        configs.append(copy.deepcopy(noise_config))

    nprocs = 2
    pool = mp.Pool(nprocs)
    pool.map(do_inversion, configs)
