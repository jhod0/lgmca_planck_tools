#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function
import pandas as pd


def load_chain(fname):
    '''
    Loads a Cobaya/CosmoMC output chain as a Pandas dataframe.
    '''
    with open(fname) as f:
        values = []
        for i, line in enumerate(f.readlines()):
            if i == 0:
                columns = [name.strip('#') for name in line.split(' ')
                           if name]
                continue
            values.append([float(n) for n in line.split(' ') if n])
    return pd.DataFrame(values, columns=columns)
