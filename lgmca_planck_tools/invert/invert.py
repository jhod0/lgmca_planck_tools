# This script will replicate the FFP8_pipeline_LGMCA notebook to construct a
# "cmb cube" and invert the raw CMB bands to separate components
# (I think/hope?)
from __future__ import division, print_function

import asyncio
import fitsio
import healpy as hp
import logging
import numpy as np
import os

from . import log
from .band_aggregation import band_aggregation
from . import planck_data


# ==================== Some Input File Paths ==================== #
# To have this in scope, use:
# $ module add lgmca_inv
# NB this does NOT work with `mrs_planck_lgmca_inv` (lower v upper case)
# that is a different command !
inv_command = 'mrs_planck_lGMCA_inv'
matmask_command = 'mrsp_matmask'

# There are 6 cmb estimation bands (cbands)
ncbands = 6

# ==================== 1st Step: Make a 'CMB Cube' =================== #

def make_CMB_cube(config, logger=log.null_logger, parallel=True):
    with log.Timer(logger, 'Step 1: CMB Cube').with_level(logging.INFO):
        # Only make a CMB cube if it doesn't already exist
        # FIXME this could be bad - if we are trying to use different set of maps and
        # end up accidentally using the same from previous analysis?
        logger.info('creating CMB cube')
        # Create the 'CMB cube'
        only_channel = config.get('only_channel')
        if only_channel is None:
            logger.debug('Constructing CMB cube of all channels')
            cmb_cube = planck_data.load_channels(config['component_fname'], logger=logger)
        else:
            logger.debug('Loading only channel: {:03}'.format(only_channel))
            cmb_cube = np.zeros((9, hp.nside2npix(2048)))
            which = np.arange(9)[planck_data.planck_freqs == only_channel][0]
            cmb_cube[which] = planck_data.load_channel(which,
                                                       fname_fmt=config['component_fname'],
                                                       logger=logger)

        # Save it
        cmb_cube_fname = os.path.join(config['output_dir'], 'input_cmb_cube_ffp8.fits')
        config['cmb_cube_fname'] = cmb_cube_fname
        logger.info('saving CMB cube to ' + cmb_cube_fname)
        cmb_cube_dirname = os.path.dirname(cmb_cube_fname)
        if not (os.path.exists(cmb_cube_dirname) and os.path.isdir(cmb_cube_dirname)):
            os.mkdir(cmb_cube_dirname)
        cube_file = fitsio.FITS(cmb_cube_fname, 'rw')
        cube_file.write(cmb_cube)
        logger.info('CMB cube saved')

    return cmb_cube, cmb_cube_fname

# ==================== 2nd Step: Do The Inversion ==================== #

def inversion(config, logger=log.null_logger):
    with log.Timer(logger, 'Step 2: inversion').with_level(logging.INFO):
        # Construct the command
        command = [inv_command, '-t6:1',
                   config['cmb_cube_fname'],
                   planck_data.tabbeam_file,
                   config['coefs_fname'].format(cmap=0),
                   config['structure_fname'],
                   os.path.join(config['output_dir'], config['band_output_name'])]

        # Run the command
        rc = asyncio.run(log.run_command(command, timeout=5e-2, logger=logger))
        if rc != 0:
            raise RuntimeError('inversion failed with exit code: {}'.format(rc))

    # TODO return something? file names of outputs?

# ==================== 3rd step: Aggregate =========================== #

def aggregate(config, logger=log.null_logger):
    with log.Timer(logger, 'Step 3: Aggregation').with_level(logging.INFO):
        logger.info('aggregating bands')
        input_fname_fmt = os.path.join(config['output_dir'], config['band_output_name'])
        input_fname_fmt = input_fname_fmt.replace('Band0', 'Band{nband}_sameres_inverted')
        band_alms, beam, cls, cmb_agg = band_aggregation(
            input_fname_fmt,
            planck_data.tabbeam[-6:],
            logger=logger
        )

        # Output directories
        data_dir = config['output_dir']
        aggregated_map_name = os.path.join(data_dir, config['name'] + '_aggregated_cmb.fits')
        aggregated_beam_name = os.path.join(data_dir, config['name'] + '_aggregated_beam.txt')
        aggregated_cls_name = os.path.join(data_dir, config['name'] + '_aggregated_cls.fits')

        config['aggregated_map_fname'] = aggregated_map_name
        config['aggregated_beam_fname'] = aggregated_beam_name
        config['aggregated_cls_fname'] = aggregated_cls_name

        # Write outputs
        logger.info('bands aggregated, saving to ' + aggregated_map_name)
        hp.write_map(aggregated_map_name, hp.reorder(cmb_agg, r2n=True), nest=True, overwrite=True)
        np.savetxt(aggregated_beam_name, beam)
        logger.info('saving Cls to ' + aggregated_cls_name)
        hp.write_cl(aggregated_cls_name, cls, overwrite=True)
        # Don't save these - waste of disc space
        # for cmap, alms in enumerate(band_alms):
        #     np.savetxt(band_name.format(cmap=cmap), alms.view(float))

    return beam, cls, cmb_agg

# ==================== 4th step: Fix for mask ======================== #

def fix_for_mask(config, logger=log.null_logger):
    with log.Timer(logger, 'Step 4: Accounting for mask').with_level(logging.INFO):
        coupling_matrix_fname = config.get('mask_coupling_matrix_fname')

        # If the coupling matrix is not specified, we need to make it
        if not coupling_matrix_fname:
            output_file = os.path.join(config['output_dir'], 'master_deconv_spectra.fits')

            maxl = config.get('matmask_maxl', 2500)

            # mrsp_matmask -t -w -v -z -l 1024 -o TQU <smap> <mask> <output_file>
            cmd = [matmask_command, '-t', '-w', '-v', '-z', '-l', str(maxl), '-o', 'TQU',
                   config['aggregated_map_fname'],
                   config['mask_fname'],
                   output_file]

            rc = log.run_command(cmd, timeout=5e-2, logger=logger)

            if rc != 0:
                raise RuntimeError(matmask_command + ' failed with exit code: ' + str(rc))

            # Outputs are:
            # master_deconv_spectra.fits
            # master_deconv_spectra_maskedmap_pspec.fits
            # master_deconv_spectra_mask_pspec.fits
            # master_deconv_spectra_mask_spec_radii.fits
            # master_deconv_spectra_coupling_matrices.fits
            if config.get('delete_tmps', True):
                tmps = [os.path.join(config['output_dir'], fname)
                        for fname in ('master_deconv_spectra.fits',
                                      'master_deconv_spectra_maskedmap_pspec.fits',
                                      'master_deconv_spectra_mask_pspec.fits',
                                      'master_deconv_spectra_mask_spec_radii.fits',
                                      'master_deconv_spectra_coupling_matrices.fits')]
                for tmp in tmps:
                    logger.debug('deleting ' + tmp)
                    os.remove(tmp)

            coupling_matrix_fname = os.path.join(config['output_dir'], 'master_deconv_spectra_coupling_matrices.fits')

        coupling_matrix = fitsio.FITS(coupling_matrix_fname)[0].read()

        # FIXME: mrsp_matmask does not produce the correct spectrum, but it does
        # produce the correct coupling matrix. we can use the coupling matrix
        # and invert it ourselves.
        # TODO: get to the bottom of this. Why does mrsp_matmask give the wrong
        # answers? Can Florent or JLS figure this out?
        mask = hp.read_map(config['mask_fname'], verbose=False)
        aggregated_cmb = hp.read_map(config['aggregated_map_fname'], verbose=False)
        # 1e6 to convert K -> mu K
        masked_powerspec = hp.anafast(mask * 1e6 * aggregated_cmb, lmax=coupling_matrix.shape[0] - 1)
        recovered_pspec = np.linalg.solve(coupling_matrix, masked_powerspec)
        ells = np.arange(recovered_pspec.size)
        ll1 = ells * (ells + 1) / (2 * np.pi) / hp.pixwin(2048, lmax=ells.size-1)**2
        hp.write_cl(os.path.join(config['output_dir'], 'mask_corrected_spectra.fits'), ll1 * recovered_pspec)

# ==================== All together ================================== #

def run_inversion(config, logger=None,
                  skip_cube=False, skip_invert=False,
                  skip_aggregation=False, skip_mask=False):
    '''
    Run the end-to-end LGMCA inversion and aggregation.

    'Config' specifies the inputs and outputs.
    '''
    # Make sure the output dir exists
    config['output_dir'] = os.path.join(config['toplevel_dir'], config['name'])
    if not os.path.exists(config['output_dir']):
        os.mkdir(config['output_dir'])

    if logger is None:
        logger = log.make_logger(config['name'],
                                 toplevel_log_file=os.path.join(config['toplevel_dir'], 'lgmca_inversion_log.txt'),
                                 log_file=os.path.join(config['output_dir'], 'log.txt'))

    timer = log.Timer(logger, 'Inversion. Outputting to directory: {}'.format(config['output_dir']))
    with timer.with_level(logging.INFO):
        # === Step 1 === #
        if skip_cube:
            cmb_cube_fname = os.path.join(config['output_dir'], 'input_cmb_cube_ffp8.fits')
            config['cmb_cube_fname'] = cmb_cube_fname
        else:
            logger.info('Skipping Step 1: CMB cube')
            cmb_cube, cmb_cube_fname = make_CMB_cube(config, logger=logger, parallel=False)

        # === Step 2 === #
        if not skip_invert:
            inversion(config, logger=logger)
            # Delete cmb cube - no need to take up disc space
            if config.get('delete_tmps', True):
                os.remove(cmb_cube_fname)
        else:
            logger.info('Skipping Step 2: Inversion')

        # === Step 3 === #
        if not skip_aggregation:
            beam, cls, cmb_agg = aggregate(config, logger=logger)
            if config.get('delete_tmps', True):
                input_fname_fmt = os.path.join(config['output_dir'], config['band_output_name'])
                input_fname_fmt = input_fname_fmt.replace('Band0', 'Band{nband}_sameres_inverted')
                for i in range(6):
                    os.remove(input_fname_fmt.format(nband=i))
        else:
            logger.info('Skipping Step 3: Aggregation')

            # Output directories
            data_dir = config['output_dir']
            aggregated_map_name = os.path.join(data_dir, config['name'] + '_aggregated_cmb.fits')
            aggregated_beam_name = os.path.join(data_dir, config['name'] + '_aggregated_beam.txt')
            aggregated_cls_name = os.path.join(data_dir, config['name'] + '_aggregated_cls.fits')

            config['aggregated_map_fname'] = aggregated_map_name
            config['aggregated_beam_fname'] = aggregated_beam_name
            config['aggregated_cls_fname'] = aggregated_cls_name

        # === Step 4 === #
        if not skip_mask:
            fix_for_mask(config, logger=logger)
        else:
            logger.info('Skipping Step 4: Accounting for mask')

    return beam, cls, cmb_agg
