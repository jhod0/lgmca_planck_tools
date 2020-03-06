# LGMCA Planck tools

This is a collection of tools for using Planck spacecraft data and simulations
with the [LGMCA](https://www.cosmostat.org/software/lgmca) component separation
algorithm.

It includes likelihoods to run MCMC chains with the
[Cobaya](https://cobaya.readthedocs.io/en/latest/) monte carlo sampling
software.

## Installation

First, clone this repository & enter the directory:

```
$ git clone git@github.com:jhod0/lgmca_planck_tools.git
$ cd lgmca_planck_tools
```

Then, install via pip:

```
$ pip install .
```

This will use the `setup.py` in this repository. If you wish to edit the code
in this repository without having to reinstall it every time, add the
`--editable` flag to `pip install`.

## LGMCA inversion

Runnable via `python -m lgmca_planck_tools.invert`. Requires the
[lgmca_inv](https://github.com/florentsureau/lgmca_inv) program to be accessible
on the PATH.

You will need to install extra data, such as LGMCA mixing weights, planck
simulations, masks, and maps.

## Likelihoods

This package includes likelihoods to be run with the Cobaya cosmological MCMC
sampler. They are tested to work with Cobaya version 2.0.5+.

Once this package is installed you can add a likelihood to any cobaya init
file, e.g.:

```yaml
likelihood:
  lgmca_planck_tools.like.FFP8Like:
    data_vector_file: /path/to/data/vector.fits
    cov_file: /path/to/data/covariance.txt
    do_rayleigh: false
    lmin: 70
    lmax: 2000
    dl: 30
```

This will load a CMB spectrum (`D_\ell = \ell (\ell + 1) C_\ell / (2 \pi)`, in
units of `\mu K^2`, stored via `healpy.write_cl()`), and a `D_\ell` covariance,
and run an MCMC chain sampling cosmological parameters to fit to the spectrum.

It will bin the input vector from `\ell = 30` to `\ell = 2000` in bins of 30,
and not attempt to account for Rayleigh scattering.

TODO: other likelihoods
