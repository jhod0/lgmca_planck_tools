#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com

from __future__ import division, print_function

from . import make_binmatrix

import numpy as np
import matplotlib.pyplot as plt

default_expected_color = 'black'
default_unbinned_color = '#337d41c0'
default_binned_color = '#b33232c0'
default_err_color = '#1f77b4'

# TODO make it possible to use this w/o covariance
class ResidualPlotter():
    def __init__(self, xs, expected_ys, cov,
                 expected_label=None,
                 binning=None,
                 expected_color=default_expected_color,
                 err_color=default_err_color):
        self.datasets = []
        self.plotters = []

        # Set color options
        self.expected_color = expected_color
        self.err_color = err_color

        # Make sure covariance makes sense
        cov = np.array(cov, dtype=np.double)
        if cov.ndim == 1:
            cov = np.diag(cov)
        elif cov.ndim > 2:
            raise ValueError('Not sure how to interpret covariance with ndim == {}'.format(cov.ndim))

        # Check consistent dimensions of inputs
        self.xs = np.copy(xs)
        if xs.shape != expected_ys.shape:
            msg = 'xs and expected ys have inconsistent shapes: {} and {}'.format(xs.shape, expected_ys.shape)
            raise ValueError(msg)
        self.expected_ys = np.copy(expected_ys)
        self.expected_label = expected_label

        if xs.shape[0] != cov.shape[0]:
            msg = 'xs and covariance have inconsistent shapes: {} and {}'.format(xs.shape, cov.shape)
            raise ValueError(msg)
        self.cov = np.copy(cov)
        self.sigmas = np.sqrt(np.diag(cov))

        # Potentially generate binning
        self.bin = False
        if binning is not None:
            xmin, xmax, dx = binning
            self.rebin(xmin, xmax, dx)

    def add_data(self, data, label=None, color=None,
                 plot_unbinned=True, plot_binned=True, binned_color=None,
                 binned_lines=False, **plot_args):
        '''
        Adds a data set to be plotted.

        data: A 1D array, same shape as `self.xs`. The data set to be plotted.
        label: The display label for this data set. Optional.
        color: Color to display this data set as. Optional.
        '''
        self.datasets.append(DataSet(self, data, label=label, color=color,
                                     plot_unbinned=plot_unbinned,
                                     plot_binned=plot_binned,
                                     binned_color=binned_color,
                                     binned_lines=binned_lines,
                                     **plot_args))
        return self.datasets[-1]

    def add_plotter(self, plotter, ax):
        '''
        Adds a residual plotter for a given axis.
        '''
        if isinstance(plotter, str):
            plotter = AxPlotter.plotter_types[plotter]
        self.plotters.append(plotter(self, ax))
        return self.plotters[-1]

    def rebin(self, minx, maxx, dx):
        self.binmatrix = make_binmatrix(minx, maxx, dx, mean=True)

        self.binned_xs = np.dot(self.binmatrix, self.xs)
        self.binned_expected_ys = np.dot(self.binmatrix, self.expected_ys)
        self.binned_cov = np.dot(self.binmatrix, np.dot(self.cov, self.binmatrix.T))
        self.binned_sigmas = np.sqrt(np.diag(self.binned_cov))
        # FIXME this is not quite right - end bin might have different dx
        self.binned_sigma_xs = [dx / 2 / np.sqrt(3)] * len(self.binned_xs)

        self.bin = True

        for ds in self.datasets:
            ds.rebin()

    def plot(self):
        '''
        Plots - executing each of the plotters.
        '''
        for p in self.plotters:
            p.plot()
        return self

    def default_plotting(self, figsize=(16, 16),
                         top='full',
                         middle='residual',
                         bottom='ratio'):
        fig, axs = plt.subplots(figsize=figsize, nrows=3, ncols=1,
                                sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

        self.add_plotter(top, axs[0])
        self.add_plotter(middle, axs[1])
        self.add_plotter(bottom, axs[2])

        return fig, axs


class DataSet:
    def __init__(self, parent, data, label, color,
                 plot_unbinned=True, plot_binned=True, binned_color=None,
                 binned_lines=False,
                 **plot_args):
        if parent.xs.shape != data.shape:
            msg = 'xs and data ys have inconsistent shapes: {} and {}'.format(parent.xs.shape, data.shape)
            raise ValueError(msg)

        self.parent = parent

        self.data = data
        self.label = label
        self.color = color
        self.plot_unbinned = plot_unbinned
        self.plot_binned = plot_binned
        self.binned_color = binned_color
        self.binned_lines = binned_lines

        self.plot_args = plot_args

        self.rebin()

    def rebin(self):
        if self.parent.bin:
            self.binned_data = np.dot(self.parent.binmatrix, self.data)
        else:
            self.binned_data = None

        return self


class AxPlotter:
    '''
    Axes plotters for use by ResidualPlotter.
    '''
    plotter_types = {}

    def __init__(self, parent, ax):
        self.parent = parent
        self.ax = ax

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AxPlotter.plotter_types[cls.plotter_name] = cls

    @classmethod
    def get_plotter(cls, name):
        return cls.plotter_types[name]

    def transform(self, datum, binned=False):
        raise NotImplementedError('only AxPlotter subclasses can use .transform()')

    def plot(self):
        # Plot expected
        self.ax.plot(self.parent.xs,
                     self.transform(self.parent.expected_ys),
                     label=self.parent.expected_label,
                     color=self.parent.expected_color)

        # Plot each data set
        for ds in self.parent.datasets:
            if ds.plot_unbinned:
                self.ax.plot(self.parent.xs, self.transform(ds.data),
                             label=ds.label, color=ds.color,
                             **ds.plot_args)
            if self.parent.bin and ds.plot_binned:
                binned_sigmas = self.transform(self.parent.binned_expected_ys + self.parent.binned_sigmas,
                                               binned=True) \
                                - self.transform(self.parent.binned_expected_ys, binned=True)
                self.ax.errorbar(self.parent.binned_xs,
                                 self.transform(ds.binned_data, binned=True),
                                 xerr=self.parent.binned_sigma_xs,
                                 yerr=binned_sigmas,
                                 label='binned {}'.format(ds.label),
                                 color=ds.binned_color,
                                 fmt='' if ds.binned_lines else 'none',
                                 **ds.plot_args)

        self.ax.fill_between(self.parent.xs,
                             self.transform(self.parent.expected_ys - self.parent.sigmas),
                             self.transform(self.parent.expected_ys + self.parent.sigmas),
                             alpha=0.3, color=self.parent.err_color,
                             label='$1\sigma$')
        self.ax.fill_between(self.parent.xs,
                             self.transform(self.parent.expected_ys - 2 * self.parent.sigmas),
                             self.transform(self.parent.expected_ys + 2 * self.parent.sigmas),
                             alpha=0.1, color=self.parent.err_color,
                             label='$2\sigma$')


class AxFullPlotter(AxPlotter):
    plotter_name = 'full'

    def transform(self, datum, binned=False):
        return datum


class AxSigmaPlotter(AxPlotter):
    plotter_name = 'sigma'

    def transform(self, datum, binned=False):
        if binned:
            mean_sigmas = np.dot(self.parent.binmatrix, self.parent.sigmas)
            return (datum - self.parent.binned_expected_ys) / mean_sigmas
        return (datum - self.parent.expected_ys) / self.parent.sigmas


class AxResidualPlotter(AxPlotter):
    plotter_name = 'residual'

    def transform(self, datum, binned=False):
        if binned:
            return datum - self.parent.binned_expected_ys
        return datum - self.parent.expected_ys


class AxRatioPlotter(AxPlotter):
    plotter_name = 'ratio'

    def transform(self, datum, binned=False):
        if binned:
            return datum / self.parent.binned_expected_ys
        return datum / self.parent.expected_ys
