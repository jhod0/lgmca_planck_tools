#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import glob
import os

from .invert import run_inversion

# Configuration:
#   - name: the name of this run.
#   - outputs_dir: outputs will be saved here (naturally), including log file.
#   - coefs_fname: the LGMCA inversion coefficients. should be usable by
#                  python `format` for cmap \in [0,6).
#   - structure_fname: a file which describes the layout of the 6 estimation bands.
#   - component_fname: the input Planck maps, usable by Planck frequency bands.
#   - band_output_name: the output name given to the inversion command.

toplevel_dir = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, os.path.pardir, os.path.pardir,
                            'lgmca_inversion_output')
pr2_data_dir = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, 'data', 'PR2_LGMCA')
component_dir = '/media/humphrey/My_Passport/Saclay/ffp8.1/mc_cmb/'
config = {'name': 'ffp8_cmb',
          'toplevel_dir': toplevel_dir,
          'coefs_fname': os.path.join(pr2_data_dir, 'FFP8_LGMCA_Band{cmap}_CoefMaps.fits'),
          'structure_fname': os.path.join(pr2_data_dir, 'params_FFP8.fits'),
          'component_fname': os.path.join(component_dir, '{0:03}', '/ffp8.1_cmb_scl_{0:03}_full_map_mc_0000.fits'),
          'band_output_name': 'lgmca_pr2coefs_Band0.fits',
          'mask_fname': os.path.join(pr2_data_dir, 'PR1_Analysis_Mask_76p.fits'),
          'mask_coupling_matrix_fname': os.path.join(pr2_data_dir, 'PR1_Analysis_Mask_76p_coupling_matrix.fits'),
          'matmask_maxl': 4000,
          'delete_tmps': False}

for i in range(5):
    config_tot = dict(config)
    config_tot['name'] = 'ffp8_all_cmb_{:04}'.format(i)
    config_tot['component_fname'] = component_dir + '{0:03}/ffp8.1_cmb_scl_{0:03}' + '_full_map_mc_{:04}.fits'.format(i)
    run_inversion(config_tot, skip_cube=True, skip_invert=True)

    for chan in [143, 217, 353, 545, 857]:
        config_only_chan = dict(config_tot)
        config_only_chan['name'] = 'ffp8_cmb_{:04}_only_{:03}'.format(i, chan)
        config_only_chan['only_channel'] = 857
        if not os.path.exists(os.path.join(toplevel_dir + config_only_chan['name'])):
            run_inversion(config_only_chan, skip_cube=True, skip_invert=True)

    cubes = glob.glob(os.path.join(toplevel_dir, '*', 'input_cmb_cube_ffp8.fits'))
    for cube in cubes:
        os.remove(cube)
