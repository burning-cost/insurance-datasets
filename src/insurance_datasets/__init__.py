"""
insurance-datasets: Synthetic UK insurance datasets with known data generating processes.

Provides ready-made datasets for testing pricing models, GLM implementations,
and actuarial workflows without needing access to real policyholder data.

Quick start
-----------
>>> from insurance_datasets import load_motor, load_home
>>> motor = load_motor(n_policies=10_000, seed=42)
>>> home = load_home(n_policies=10_000, seed=42)
"""

from insurance_datasets.motor import (
    TRUE_FREQ_PARAMS as MOTOR_TRUE_FREQ_PARAMS,
    TRUE_SEV_PARAMS as MOTOR_TRUE_SEV_PARAMS,
    load_motor,
)
from insurance_datasets.home import (
    TRUE_FREQ_PARAMS as HOME_TRUE_FREQ_PARAMS,
    TRUE_SEV_PARAMS as HOME_TRUE_SEV_PARAMS,
    load_home,
)

# Convenience: also export the motor params under their original names for
# code that just does `from insurance_datasets import TRUE_FREQ_PARAMS`
TRUE_FREQ_PARAMS = MOTOR_TRUE_FREQ_PARAMS
TRUE_SEV_PARAMS = MOTOR_TRUE_SEV_PARAMS

__all__ = [
    "load_motor",
    "load_home",
    "TRUE_FREQ_PARAMS",
    "TRUE_SEV_PARAMS",
    "MOTOR_TRUE_FREQ_PARAMS",
    "MOTOR_TRUE_SEV_PARAMS",
    "HOME_TRUE_FREQ_PARAMS",
    "HOME_TRUE_SEV_PARAMS",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-datasets")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
