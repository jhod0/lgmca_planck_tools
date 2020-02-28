# LGMCA Planck tools

This is a collection of tools for using Planck spacecraft data and simulations
with the [LGMCA](https://www.cosmostat.org/software/lgmca) component separation
algorithm.

It includes likelihoods to run MCMC chains with the
[Cobaya](https://cobaya.readthedocs.io/en/latest/) monte carlo sampling
software.

## LGMCA inversion

Runnable via `python -m lgmca_planck_tools.invert`. Requires the
[lgmca_inv](https://github.com/florentsureau/lgmca_inv) program to be accessible
on the PATH.

You will need to install extra data, such as LGMCA mixing weights, planck
simulations, masks, and maps.

## Likelihoods

TODO
