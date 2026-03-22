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

# Backwards-compat aliases: these silently point to motor params.
# Deprecated: import MOTOR_TRUE_FREQ_PARAMS / HOME_TRUE_FREQ_PARAMS explicitly.
def __getattr__(name: str):
    import warnings as _warnings
    _deprecated = {
        "TRUE_FREQ_PARAMS": "MOTOR_TRUE_FREQ_PARAMS",
        "TRUE_SEV_PARAMS": "MOTOR_TRUE_SEV_PARAMS",
    }
    if name in _deprecated:
        preferred = _deprecated[name]
        _warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            f"It silently aliases the motor dataset parameters. "
            f"Use {preferred} or HOME_TRUE_FREQ_PARAMS / HOME_TRUE_SEV_PARAMS explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return MOTOR_TRUE_FREQ_PARAMS if "FREQ" in name else MOTOR_TRUE_SEV_PARAMS
    raise AttributeError(f"module 'insurance_datasets' has no attribute {name!r}")

# Note: TRUE_FREQ_PARAMS and TRUE_SEV_PARAMS are intentionally NOT assigned
# as module-level variables here. They are served via __getattr__ above so
# that a DeprecationWarning is raised when code accesses them.
# Static type checkers will see them via __all__.

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
