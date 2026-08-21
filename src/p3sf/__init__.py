"""Spectroscopic selection function of changing-look AGN.

Estimates the conditional classification efficiency P(C=1 | T, S/N, f_host, R, z,
survey, D): the probability that a genuine transition T receives a changing-look
label under given observing conditions and a given published criterion D.

This is not a survey selection function. See 00_admin/PROTOCOL_DEVIATIONS.md.
"""

from p3sf.config import Config, load_config, project_root

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "project_root", "__version__"]
