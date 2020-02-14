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
from ..like import gen_ffp8_1_like


def make_info(freq, realization,
              data_covname,
              param_covname=None,
              output_dir='chains'):
    # Read in the default cobaya YAML for ffp8.1
    with open(os.path.join(os.path.dirname(__file__),
                           'cobaya_ffp8_1.yaml')) as f:
        info = yaml.safe_load(f)

    # Create our likelihood
    my_like = gen_ffp8_1_like(freq, realization, data_covname)

    # Add our likelihood to the info
    my_like_name = 'ffp8_1_raw_{:03}_{:04}'.format(freq, realization)
    info['likelihood'] = {my_like_name: {'external': my_like,
                                         'speed': 100}}
    info['output'] = os.path.join(output_dir, my_like_name, my_like_name)

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

    parser.add_argument('channel', type=int,
                        help='Planck frequency channel to use. One of: {}'.format(list(planck_freqs)))
    parser.add_argument('realization', type=int,
                        help='FFP8.1 realization number, [0-99]')
    parser.add_argument('--output', '-o', dest='output_dir', default='chains',
                        help='Output directory of the chain')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='If this flag is added, the script prints the ' \
                             'yaml config passed to Cobaya and does not run ' \
                             'any chain.')
    parser.add_argument('--data-cov', dest='data_cov', default=data_cov_fname,
                        help='Path to data covariance matrix. By default, ' \
                             'the expected cosmic variance for the FFP8.1 ' \
                             'cosmology')
    parser.add_argument('--param-cov', dest='param_cov', default=param_cov_fname,
                        help='Path to parameter covariance matrix, i.e. the ' \
                             'proposal matrix. By default it is the ' \
                             'resulting proposal covariance learned from a ' \
                             'MCMC run on the 217 GHz channel.')


    args = parser.parse_args()

    if args.channel not in planck_freqs:
        raise ValueError(
            'Invalid channel {}, should be one of: {}'.format(args.channel,
                                                              list(planck_freqs))
        )
    if args.realization < 0 or args.realization >= 100:
        raise ValueError(
            'Invalid realization {}, should be in 0-99'.format(args.realization)
        )

    info = make_info(args.channel, args.realization, args.data_cov,
                     param_covname=args.param_cov,
                     output_dir=args.output_dir)
    if args.dry_run:
        yaml.dump(info, sys.stdout)
    else:
        run(info)
