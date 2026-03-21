"""
Synthetic UK household insurance dataset.

Generates realistic household policy and claims data for testing and
experimentation. The data generating process (DGP) uses known true parameters
so fitted GLMs can be validated against the ground truth.

Design decisions
----------------
UK household insurance splits into two covers: buildings and contents. This
dataset models them jointly on a single policy record. Key rating factors are:

- Property value (sum insured): log-normally distributed, anchored to UK
  regional house price distributions. Higher values raise both severity and,
  mildly, frequency (more to go wrong in a bigger house).
- Construction type: standard (brick/tile), non-standard (flat roof, timber
  frame), and listed/historic. Non-standard construction has a substantial
  frequency load.
- Flood zone: Environment Agency flood risk bands. Zone 3 (high risk) is
  the most significant frequency driver for buildings claims.
- Subsidence risk: highly correlated with London clay and other shrinkable
  soils. Treated as a binary flag for simplicity.
- Contents value: independent log-normal, correlated weakly with property
  value. Main driver of contents claim severity.
- Security level: basic (no alarm), standard (BS-approved alarm), or enhanced
  (monitored alarm + window locks). Reduces theft frequency.

The true frequency model is::

    log(lambda) = log(exposure) + alpha_0
                + alpha_property_value * log(property_value / 250000)
                + alpha_construction * construction_effect
                + alpha_flood * flood_effect
                + alpha_subsidence * is_subsidence_risk
                + alpha_security * security_effect

The true severity model is::

    log(mu) = delta_0
            + delta_property_value * log(property_value / 250000)
            + delta_flood * (flood_zone == 'Zone 3')
            + delta_contents * log(contents_value / 30000)

These parameters are exported as ``TRUE_FREQ_PARAMS`` and ``TRUE_SEV_PARAMS``
so test code can check GLM recovery.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final, Union

import numpy as np
import pandas as pd

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

# ---------------------------------------------------------------------------
# True DGP parameters
# ---------------------------------------------------------------------------

TRUE_FREQ_PARAMS: Final[dict[str, float]] = {
    "intercept": -2.8,                  # log baseline frequency; ~6% per year
    "property_value_log": 0.18,         # per unit of log(property_value / 250k)
    "construction_non_standard": 0.40,  # flat roof, timber frame etc.
    "construction_listed": 0.25,        # listed/historic buildings
    "flood_zone_2": 0.30,               # moderate flood risk
    "flood_zone_3": 0.85,               # high flood risk
    "subsidence_risk": 0.55,            # shrinkable clay soils
    "security_standard": -0.10,         # BS-approved alarm vs. no alarm
    "security_enhanced": -0.25,         # monitored alarm + window locks
}

TRUE_SEV_PARAMS: Final[dict[str, float]] = {
    "intercept": 8.1,                   # log baseline severity ~£3,300
    "property_value_log": 0.35,         # higher value = higher cost to repair
    "flood_zone_3": 0.45,               # flood claims are expensive
    "contents_value_log": 0.22,         # per unit of log(contents_value / 30k)
}

CONSTRUCTION_TYPES: Final[list[str]] = ["Standard", "Non-Standard", "Listed"]
CONSTRUCTION_PROBS: Final[list[float]] = [0.82, 0.13, 0.05]

FLOOD_ZONES: Final[list[str]] = ["Zone 1", "Zone 2", "Zone 3"]
# Environment Agency estimates: ~75% Zone 1, ~15% Zone 2, ~10% Zone 3
FLOOD_PROBS: Final[list[float]] = [0.75, 0.15, 0.10]

SECURITY_LEVELS: Final[list[str]] = ["Basic", "Standard", "Enhanced"]
SECURITY_PROBS: Final[list[float]] = [0.30, 0.50, 0.20]

# UK regions for property value calibration (broadly ONS regional groupings)
REGIONS: Final[list[str]] = [
    "London", "South East", "East of England", "South West",
    "East Midlands", "West Midlands", "Yorkshire", "North West",
    "North East", "Wales", "Scotland", "Northern Ireland",
]
# Mean property values (£) by region — loosely based on UK HPI 2022
REGION_MEAN_VALUES: Final[dict[str, float]] = {
    "London": 520_000,
    "South East": 380_000,
    "East of England": 330_000,
    "South West": 300_000,
    "East Midlands": 230_000,
    "West Midlands": 235_000,
    "Yorkshire": 195_000,
    "North West": 205_000,
    "North East": 160_000,
    "Wales": 185_000,
    "Scotland": 175_000,
    "Northern Ireland": 155_000,
}
# Approximate share of UK housing stock by region
REGION_PROBS: Final[list[float]] = [
    0.13, 0.14, 0.10, 0.09, 0.08, 0.09, 0.08, 0.11, 0.04, 0.05, 0.07, 0.02
]


def _construction_effect(construction: np.ndarray) -> np.ndarray:
    """
    Map construction type strings to log-frequency effects.

    Parameters
    ----------
    construction : np.ndarray of str
        Values from ``CONSTRUCTION_TYPES``.

    Returns
    -------
    np.ndarray of float
    """
    effect = np.zeros(len(construction))
    effect[construction == "Non-Standard"] = TRUE_FREQ_PARAMS["construction_non_standard"]
    effect[construction == "Listed"] = TRUE_FREQ_PARAMS["construction_listed"]
    return effect


def _flood_freq_effect(flood_zone: np.ndarray) -> np.ndarray:
    """
    Map flood zone strings to log-frequency effects.

    Parameters
    ----------
    flood_zone : np.ndarray of str
        Values from ``FLOOD_ZONES``.

    Returns
    -------
    np.ndarray of float
    """
    effect = np.zeros(len(flood_zone))
    effect[flood_zone == "Zone 2"] = TRUE_FREQ_PARAMS["flood_zone_2"]
    effect[flood_zone == "Zone 3"] = TRUE_FREQ_PARAMS["flood_zone_3"]
    return effect


def _security_effect(security: np.ndarray) -> np.ndarray:
    """
    Map security level strings to log-frequency effects (reductions).

    Parameters
    ----------
    security : np.ndarray of str
        Values from ``SECURITY_LEVELS``.

    Returns
    -------
    np.ndarray of float
    """
    effect = np.zeros(len(security))
    effect[security == "Standard"] = TRUE_FREQ_PARAMS["security_standard"]
    effect[security == "Enhanced"] = TRUE_FREQ_PARAMS["security_enhanced"]
    return effect


def _generate_policies(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate the household policy characteristics table.

    Policies span a 5-year window (2019-2023). About 5% cancel early
    (household insurance has lower cancellation rates than motor).

    Parameters
    ----------
    n : int
        Number of policies to generate.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    pd.DataFrame
        Policy characteristics without claims or exposure columns.
    """
    base_date = date(2019, 1, 1)
    total_days = 5 * 365

    inception_days = rng.integers(0, total_days, size=n)
    inception_dates = [base_date + timedelta(days=int(d)) for d in inception_days]

    is_cancellation = rng.random(n) < 0.05
    cancel_fraction = rng.uniform(0.10, 0.90, n)

    expiry_dates = []
    for i, inc in enumerate(inception_dates):
        try:
            full_expiry = date(inc.year + 1, inc.month, inc.day)
        except ValueError:
            full_expiry = date(inc.year + 1, inc.month, inc.day - 1)

        if is_cancellation[i]:
            term_days = (full_expiry - inc).days
            actual_days = max(1, int(term_days * cancel_fraction[i]))
            expiry_dates.append(inc + timedelta(days=actual_days))
        else:
            expiry_dates.append(full_expiry)

    # Region and property value
    regions = rng.choice(REGIONS, size=n, p=REGION_PROBS)
    region_means = np.array([REGION_MEAN_VALUES[r] for r in regions])
    # Log-normal around regional mean, sigma ~0.4 gives realistic spread
    log_means = np.log(region_means) - 0.5 * 0.4 ** 2  # correct for log-normal bias
    property_value = np.clip(
        np.exp(rng.normal(log_means, 0.4)).astype(int),
        50_000, 2_000_000
    )

    # Contents value: weakly correlated with property value
    # Mean ~£30k, modest spread
    contents_log_mean = 10.0 + 0.2 * (np.log(property_value) - np.log(250_000))
    contents_value = np.clip(
        np.exp(rng.normal(contents_log_mean, 0.45)).astype(int),
        5_000, 250_000
    )

    construction = rng.choice(CONSTRUCTION_TYPES, size=n, p=CONSTRUCTION_PROBS)

    # Flood zone: mildly correlated with region (London/Thames higher)
    flood_zone_probs = np.tile(FLOOD_PROBS, (n, 1))
    london_mask = regions == "London"
    flood_zone_probs[london_mask] = [0.60, 0.25, 0.15]
    flood_zone = np.array([
        rng.choice(FLOOD_ZONES, p=flood_zone_probs[i]) for i in range(n)
    ])

    # Subsidence: London clay and Midlands soils most susceptible
    subsidence_base = np.where(
        np.isin(regions, ["London", "East Midlands", "West Midlands"]), 0.12, 0.04
    )
    is_subsidence_risk = rng.random(n) < subsidence_base

    security = rng.choice(SECURITY_LEVELS, size=n, p=SECURITY_PROBS)

    # Number of bedrooms: drives contents value indirectly
    bedrooms = np.clip(
        rng.choice([1, 2, 3, 4, 5], size=n, p=[0.12, 0.28, 0.35, 0.18, 0.07]),
        1, 5
    )

    # Property age band
    property_age_band = rng.choice(
        ["Pre-1900", "1900-1945", "1945-1980", "1980-2000", "Post-2000"],
        size=n,
        p=[0.12, 0.20, 0.30, 0.22, 0.16]
    )

    return pd.DataFrame({
        "inception_date": inception_dates,
        "expiry_date": expiry_dates,
        "region": regions,
        "property_value": property_value,
        "contents_value": contents_value,
        "construction_type": construction,
        "flood_zone": flood_zone,
        "is_subsidence_risk": is_subsidence_risk,
        "security_level": security,
        "bedrooms": bedrooms,
        "property_age_band": property_age_band,
    })


def _calculate_earned_exposure(df: pd.DataFrame) -> pd.Series:
    """
    Calculate earned exposure in years for each policy row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``inception_date`` and ``expiry_date`` columns.

    Returns
    -------
    pd.Series of float
        Exposure in policy years, clipped at zero.
    """
    days = (
        pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["inception_date"])
    ).dt.days
    return (days / 365.25).clip(lower=0.0)


def _generate_claims(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.Series, pd.Series]:
    """
    Generate claim counts and incurred amounts from the true DGP.

    Frequency model: Poisson with log-linear predictor. See ``TRUE_FREQ_PARAMS``.
    Severity model: Gamma with log-linear predictor. See ``TRUE_SEV_PARAMS``.
    Gamma shape = 1.5, giving a coefficient of variation of ~0.82 (household
    claims are more variable than motor — big flood events skew the distribution).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all rating factor columns plus ``exposure``.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    claim_count : pd.Series of int
    incurred : pd.Series of float
        Zero where ``claim_count == 0``.
    """
    n = len(df)

    property_value_scaled = np.log(df["property_value"].values / 250_000)

    log_lambda = (
        TRUE_FREQ_PARAMS["intercept"]
        + TRUE_FREQ_PARAMS["property_value_log"] * property_value_scaled
        + _construction_effect(df["construction_type"].values)
        + _flood_freq_effect(df["flood_zone"].values)
        + TRUE_FREQ_PARAMS["subsidence_risk"] * df["is_subsidence_risk"].values.astype(float)
        + _security_effect(df["security_level"].values)
    )

    exposure = df["exposure"].values
    log_lambda += np.log(np.clip(exposure, 1e-6, None))

    lambda_vals = np.exp(log_lambda)
    claim_count = rng.poisson(lambda_vals)

    contents_value_scaled = np.log(df["contents_value"].values / 30_000)

    log_mu = (
        TRUE_SEV_PARAMS["intercept"]
        + TRUE_SEV_PARAMS["property_value_log"] * property_value_scaled
        + TRUE_SEV_PARAMS["flood_zone_3"] * (df["flood_zone"].values == "Zone 3").astype(float)
        + TRUE_SEV_PARAMS["contents_value_log"] * contents_value_scaled
    )

    gamma_shape = 1.5
    gamma_mean = np.exp(log_mu)
    gamma_scale = gamma_mean / gamma_shape

    has_claims = claim_count > 0
    incurred = np.zeros(n)

    if has_claims.any():
        for i in np.where(has_claims)[0]:
            per_claim = rng.gamma(
                shape=gamma_shape,
                scale=gamma_scale[i],
                size=int(claim_count[i])
            )
            incurred[i] = per_claim.sum()

    return pd.Series(claim_count, dtype=int), pd.Series(incurred, dtype=float)


def load_home(
    n_policies: int = 50_000,
    seed: int = 42,
    polars: bool = False,
) -> "Union[pd.DataFrame, pl.DataFrame]":
    """
    Load a synthetic UK household insurance dataset.

    Generates ``n_policies`` rows with realistic UK household insurance
    characteristics and simulated claims from a known data generating process.
    Both buildings and contents cover are represented on a single policy row.
    The true parameters (see ``TRUE_FREQ_PARAMS`` and ``TRUE_SEV_PARAMS``)
    allow GLM coefficient recovery testing.

    Parameters
    ----------
    n_policies : int
        Number of policies to generate. Default 50,000 gives stable GLM
        estimates. Use 5,000-10,000 for quick tests.
    seed : int
        Random seed for reproducibility.
    polars : bool
        If True, return a polars DataFrame instead of pandas. Polars is an
        optional dependency — install it with ``pip install polars`` if needed.
        Default False.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        One row per policy with columns:

        - ``policy_id`` : int — sequential identifier
        - ``inception_date`` : date — policy start
        - ``expiry_date`` : date — policy end
        - ``inception_year`` : int — calendar year of inception (for cohort splits)
        - ``region`` : str — UK region (ONS groupings)
        - ``property_value`` : int — buildings sum insured (£)
        - ``contents_value`` : int — contents sum insured (£)
        - ``construction_type`` : str — 'Standard', 'Non-Standard', or 'Listed'
        - ``flood_zone`` : str — Environment Agency zone: 'Zone 1', 'Zone 2', 'Zone 3'
        - ``is_subsidence_risk`` : bool — high subsidence area flag
        - ``security_level`` : str — 'Basic', 'Standard', or 'Enhanced'
        - ``bedrooms`` : int — number of bedrooms (1-5)
        - ``property_age_band`` : str — construction era band
        - ``claim_count`` : int — number of claims in period
        - ``incurred`` : float — total incurred cost (0.0 if no claims)
        - ``exposure`` : float — earned years (< 1.0 for cancellations)

    Examples
    --------
    >>> from insurance_datasets import load_home
    >>> df = load_home(n_policies=10_000, seed=0)
    >>> df.shape
    (10000, 16)
    >>> df["claim_count"].sum() / df["exposure"].sum()  # claim frequency
    # approximately 0.05-0.07
    >>> df_pl = load_home(n_policies=1_000, seed=0, polars=True)
    >>> type(df_pl).__name__
    'DataFrame'

    Notes
    -----
    Claims represent combined buildings and contents losses. The peril mix
    includes escape of water, accidental damage, theft, and subsidence.
    The DGP does not model individual perils separately.

    There are no missing values by design.
    """
    if polars and not _POLARS_AVAILABLE:
        raise ImportError("Install polars: pip install polars")
    rng = np.random.default_rng(seed)

    df = _generate_policies(n_policies, rng)
    df["exposure"] = _calculate_earned_exposure(df)
    df["inception_year"] = pd.to_datetime(df["inception_date"]).dt.year
    df["claim_count"], df["incurred"] = _generate_claims(df, rng)
    df.insert(0, "policy_id", np.arange(1, n_policies + 1))

    column_order = [
        "policy_id",
        "inception_date",
        "expiry_date",
        "inception_year",
        "region",
        "property_value",
        "contents_value",
        "construction_type",
        "flood_zone",
        "is_subsidence_risk",
        "security_level",
        "bedrooms",
        "property_age_band",
        "claim_count",
        "incurred",
        "exposure",
    ]
    df = df[column_order].copy()

    df["policy_id"] = df["policy_id"].astype(int)
    df["inception_date"] = pd.to_datetime(df["inception_date"]).dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["inception_year"] = df["inception_year"].astype(int)
    df["property_value"] = df["property_value"].astype(int)
    df["contents_value"] = df["contents_value"].astype(int)
    df["is_subsidence_risk"] = df["is_subsidence_risk"].astype(bool)
    df["bedrooms"] = df["bedrooms"].astype(int)
    df["claim_count"] = df["claim_count"].astype(int)
    df["incurred"] = df["incurred"].astype(float)
    df["exposure"] = df["exposure"].astype(float)

    df = df.reset_index(drop=True)
    if polars:
        return pl.from_pandas(df)
    return df
