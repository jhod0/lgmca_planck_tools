#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import numpy as np
from ..binmat import make_binmatrix

def test_binmatrix_simple():
    bmat_1 = make_binmatrix(2, 5, 2)
    expected_1 = np.array([[0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1]])
    assert np.all(bmat_1 == expected_1)

    bmat_2 = make_binmatrix(1, 5, 2)
    expected_2 = np.array([[0, 1, 1, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
    assert np.all(bmat_2 == expected_2)

    bmat_3 = make_binmatrix(0, 4, 2)
    expected_3 = np.array([[1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 1]])
    assert np.all(bmat_3 == expected_3)
