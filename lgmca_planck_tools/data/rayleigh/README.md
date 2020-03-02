# Rayleigh Scattering

This directory contains a template of the effect of Rayleigh scattering on the
CMB at the cosmology used in the Planck Full Focal Plane 8.1 (FFP8.1)
simulations.

It is derived from the `rayleigh` branch of
[CAMB](https://github.com/cmbant/CAMB). The version used here is a fork to
fix some compilation errors, found
[here](https://github.com/jhod0/CAMB/tree/rayleigh).

The inputs used to generate the rayleigh templates are `ffp8_1_rayleigh/*.ini`.
Rayleigh scattering is a frequency-dependent effect, modeled as a Taylor
expansion of orders `\nu^4`, `\nu^6`, and `\nu^8`. Three `.ini` files are
used to give the template of each power, and one to give the total Rayleigh
effect amplitude.
