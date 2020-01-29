#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import numpy as np

# Central frequencies of each planck observation band, in GHz
planck_freqs = np.array([30, 44, 70, 100, 143, 217, 353, 545, 857])
planck_freqs.flags.writeable = False

# (JOD: IMO this is pathologically bad but the last 2 bands are in different units.
#       maybe there are good reasons for this that I don't know)
# Correction factors for MJy/sr -> KCMB
# MJy/sr / KCMB  might need to be updated
cfact_545_857 = np.array([58.035560, 2.2681256])
cfact_545_857.flags.writeable = False

# Doppler biases by frequency band. From FFP8 paper, table 4
ffp8_doppler_adjs = np.array([1.05, 1.1, 1.25, 1.51, 1.96, 3.07, 5.38, 8.82, 14.2])
ffp8_doppler_adjs.flags.writeable = False

# FFP8 dipole direction and magnitude
ffp8_theta_dipole = np.pi/2 - (48.4 * np.pi / 180)
ffp8_phi_dipole = 264.4 * np.pi / 180
ffp8_dipole_magnitude = 0.00123
