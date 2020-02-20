#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

import argparse
from cobaya.run import run
import os
import sys
import yaml

from ..planck import planck_freqs
from ..like import gen_ffp8_1_like, gen_lgmca_like


def make_info(freq, realization,
              data_covname,
              binning,
              param_covname=None,
              output_dir='chains',
              resume=False):
    # Read in the default cobaya YAML for ffp8.1
    with open(os.path.join(os.path.dirname(__file__),
                           'cobaya_ffp8_1.yaml')) as f:
        info = yaml.safe_load(f)

    lmin, lmax, dl = binning
    # Create our likelihood
    if freq.lower() == 'lgmca':
        my_like_name = 'ffp8_1_raw_lgmca_{:04}'.format(realization)
        try:
            lgmca_dir = os.environ['LGMCA_OUTPUT_DIR']
        except KeyError:
            print("Don't know where to find LGMCA outputs - set the " \
                  'environment variable LGMCA_OUTPUT_DIR')
            raise
        lgmca_file = os.path.join(lgmca_dir,
                                  'ffp8_mc_cmb_{:04}'.format(realization),
                                  'FFP8_v1_aggregated_cls.fits')
        my_like = gen_lgmca_like(lgmca_file, data_covname,
                                 lmin=lmin, lmax=lmax, delta_ell=dl)
    else:
        my_like_name = 'ffp8_1_raw_{:03}_{:04}'.format(int(freq), realization)
        my_like = gen_ffp8_1_like(int(freq), realization, data_covname,
                                  lmin=lmin, lmax=lmax, delta_ell=dl)

    # Add our likelihood to the info
    info['likelihood'] = {my_like_name: {'external': my_like,
                                         'speed': 100}}
    info['output'] = os.path.join(output_dir, my_like_name, my_like_name)
    info['resume'] = resume

    # If we have a proposal matrix, add it
    if param_covname:
        info['sampler']['mcmc']['covmat'] = param_covname

    return info


if __name__ == '__main__':
    data_cov_fname = os.path.join(os.path.dirname(__file__),
                                  '..', 'data', 'covariances', 'data',
                                  'ffp8_1_expected_cosmicvar.txt')
    param_cov_fname = os.path.join(os.path.dirname(__file__),
                                   '..', 'data', 'covariances', 'param',
                                   'ffp8_1_217_0001.covmat')

    parser = argparse.ArgumentParser(description='Run a Cobaya chain for FFP8.1')

    parser.add_argument('channel', type=str,
                        help='Planck frequency channel to use, or LGMCA output. ' \
                             'Either lgmca or one of: {}'.format(list(planck_freqs)))
    parser.add_argument('realization', type=int,
                        help='FFP8.1 realization number, [0-99]')
    parser.add_argument('--output', '-o', dest='output_dir', default='chains',
                        help='Output directory of the chain')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='If this flag is added, the script prints the ' \
                             'yaml config passed to Cobaya and does not run ' \
                             'any chain.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Whether this chain is resuming a previous chain.')
    parser.add_argument('--data-cov', dest='data_cov', default=data_cov_fname,
                        help='Path to data covariance matrix. By default, ' \
                             'the expected cosmic variance for the FFP8.1 ' \
                             'cosmology')
    parser.add_argument('--param-cov', dest='param_cov', default=param_cov_fname,
                        help='Path to parameter covariance matrix, i.e. the ' \
                             'proposal matrix. By default it is the ' \
                             'resulting proposal covariance learned from a ' \
                             'MCMC run on the 217 GHz channel.')

    parser.add_argument('--lmin', type=int, default=70,
                        help='Minimum multipole \ell used in the analysis')
    parser.add_argument('--lmax', type=int, default=2000,
                        help='Maximum multipole \ell used in the analysis')
    parser.add_argument('--delta-ell', type=int, default=30,
                        help='Delta \ell, the size of binning to use on the ' \
                             'data vector in the analysis')


    args = parser.parse_args()

    if not ((args.channel.lower() == 'lgmca') or
            (int(args.channel) in planck_freqs)):
        raise ValueError(
            'Invalid channel {}, should be one of: {}'.format(args.channel,
                                                              list(planck_freqs))
        )
    if args.realization < 0 or args.realization >= 100:
        raise ValueError(
            'Invalid realization {}, should be in 0-99'.format(args.realization)
        )

    print('Using binning', args.lmin, args.lmax, args.delta_ell)
    info = make_info(args.channel, args.realization, args.data_cov,
                     binning=(args.lmin, args.lmax, args.delta_ell),
                     param_covname=args.param_cov,
                     output_dir=args.output_dir,
                     resume=args.resume)
    if args.dry_run:
        yaml.dump(info, sys.stdout)
    else:
        run(info)
