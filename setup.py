#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function

from distutils.core import setup

data_dirs = ['data/covariances/data/*',
             'data/covariances/param/*',
             'data/ffp8/rimos/*.fits',
             'data/rimos/*.fits',
             'data/rimos/BeamWf_HFI_R3.01/*.fits',
             'data/rayleigh/ffp8_1_rayleigh/*.ini',
             'data/rayleigh/ffp8_1_rayleigh/*.dat',
             'data/rayleigh/ffp8_1_rayleigh/Rayleigh/*',
            ]

setup(name='lgmca_planck_tools',
      version='0.1',
      description='Tools for using Planck data and simulations with the ' \
                  'LGMCA component separation algorithm',
      author="Jackson O'Donnell",
      author_email='jacksonhodonnell@gmail.com',
      packages=[
          'lgmca_planck_tools',
          'lgmca_planck_tools.cobaya',
          'lgmca_planck_tools.like',
#           'lgmca_planck_tools.like.ffp8_like',
#           'lgmca_planck_tools.like.lgmca_like',
          'lgmca_planck_tools.planck',
      ],
      package_data={'lgmca_planck_tools': data_dirs,
                    'lgmca_planck_tools.like': ['*.yaml']},
      install_requires=[
          'camb',
          'cobaya',
          'fitsio',
          'healpy',
          'numpy',
          'matplotlib',
      ])
