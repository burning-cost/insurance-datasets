"""
Tests for insurance_datasets.home.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_datasets import (
    HOME_TRUE_FREQ_PARAMS,
    HOME_TRUE_SEV_PARAMS,
    load_home,
)
from insurance_datasets.home import (
    CONSTRUCTION_TYPES,
    FLOOD_ZONES,
    REGIONS,
    SECURITY_LEVELS,
)


# ---------------------------------------------------------------------------
# Basic shape and dtype tests
# ---------------------------------------------------------------------------


def test_load_home_default_shape() -> None:
    df = load_home(n_policies=1_000, seed=0)
    assert df.shape == (1_000, 16)


def test_load_home_column_names() -> None:
    df = load_home(n_policies=100, seed=0)
    expected = [
        "policy_id", "inception_date", "expiry_date", "inception_year",
        "region", "property_value", "contents_value", "construction_type",
        "flood_zone", "is_subsidence_risk", "security_level", "bedrooms",
        "property_age_band", "claim_count", "incurred", "exposure",
    ]
    assert list(df.columns) == expected


def test_load_home_dtypes() -> None:
    df = load_home(n_policies=100, seed=0)
    assert df["is_subsidence_risk"].dtype == np.dtype("bool")
    assert df["incurred"].dtype == np.dtype("float64")
    assert df["exposure"].dtype == np.dtype("float64")
    assert df["property_value"].dtype in (np.dtype("int64"), np.dtype("int32"))


def test_no_missing_values() -> None:
    df = load_home(n_policies=1_000, seed=0)
    assert df.isnull().sum().sum() == 0


def test_policy_id_sequential() -> None:
    n = 300
    df = load_home(n_policies=n, seed=0)
    assert list(df["policy_id"]) == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# Domain constraint tests
# ---------------------------------------------------------------------------


def test_property_value_range() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert df["property_value"].min() >= 50_000
    assert df["property_value"].max() <= 2_000_000


def test_contents_value_range() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert df["contents_value"].min() >= 5_000
    assert df["contents_value"].max() <= 250_000


def test_construction_type_valid() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert set(df["construction_type"].unique()).issubset(set(CONSTRUCTION_TYPES))


def test_flood_zone_valid() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert set(df["flood_zone"].unique()).issubset(set(FLOOD_ZONES))


def test_security_level_valid() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert set(df["security_level"].unique()).issubset(set(SECURITY_LEVELS))


def test_region_valid() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert set(df["region"].unique()).issubset(set(REGIONS))


def test_bedrooms_range() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert df["bedrooms"].min() >= 1
    assert df["bedrooms"].max() <= 5


def test_exposure_positive() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert (df["exposure"] > 0).all()


def test_exposure_at_most_one_year() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert (df["exposure"] <= 1.01).all()


def test_incurred_non_negative() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert (df["incurred"] >= 0).all()


def test_incurred_zero_when_no_claim() -> None:
    df = load_home(n_policies=5_000, seed=2)
    no_claim = df["claim_count"] == 0
    assert (df.loc[no_claim, "incurred"] == 0.0).all()


def test_incurred_positive_when_claim() -> None:
    df = load_home(n_policies=5_000, seed=2)
    has_claim = df["claim_count"] > 0
    assert (df.loc[has_claim, "incurred"] > 0).all()


def test_inception_year_range() -> None:
    df = load_home(n_policies=2_000, seed=1)
    assert df["inception_year"].min() >= 2019
    assert df["inception_year"].max() <= 2023


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed() -> None:
    df1 = load_home(n_policies=500, seed=77)
    df2 = load_home(n_policies=500, seed=77)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ() -> None:
    df1 = load_home(n_policies=500, seed=20)
    df2 = load_home(n_policies=500, seed=21)
    assert df1["incurred"].sum() != df2["incurred"].sum()


# ---------------------------------------------------------------------------
# Statistical plausibility tests
# ---------------------------------------------------------------------------


def test_claim_frequency_plausible() -> None:
    """Portfolio average claim frequency should be in the 3-10% range."""
    df = load_home(n_policies=10_000, seed=42)
    freq = df["claim_count"].sum() / df["exposure"].sum()
    assert 0.03 < freq < 0.10, f"Claim frequency {freq:.4f} outside expected range"


def test_mean_severity_plausible() -> None:
    """Mean claim severity should be in the £1,500-£8,000 range."""
    df = load_home(n_policies=10_000, seed=42)
    claims = df[df["claim_count"] > 0]
    mean_sev = claims["incurred"].sum() / claims["claim_count"].sum()
    assert 1_500 < mean_sev < 8_000, f"Mean severity £{mean_sev:.0f} outside expected range"


def test_flood_zone_3_higher_frequency() -> None:
    """Zone 3 should have materially higher claim frequency than Zone 1."""
    df = load_home(n_policies=20_000, seed=42)
    z1 = df[df["flood_zone"] == "Zone 1"]
    z3 = df[df["flood_zone"] == "Zone 3"]
    freq_z1 = z1["claim_count"].sum() / z1["exposure"].sum()
    freq_z3 = z3["claim_count"].sum() / z3["exposure"].sum()
    assert freq_z3 > freq_z1 * 1.5, (
        f"Zone 3 frequency {freq_z3:.4f} not sufficiently above Zone 1 {freq_z1:.4f}"
    )


def test_non_standard_construction_higher_frequency() -> None:
    """Non-standard construction should have higher frequency than standard."""
    df = load_home(n_policies=20_000, seed=42)
    std = df[df["construction_type"] == "Standard"]
    nonstd = df[df["construction_type"] == "Non-Standard"]
    freq_std = std["claim_count"].sum() / std["exposure"].sum()
    freq_nonstd = nonstd["claim_count"].sum() / nonstd["exposure"].sum()
    assert freq_nonstd > freq_std, (
        f"Non-standard frequency {freq_nonstd:.4f} should exceed standard {freq_std:.4f}"
    )


def test_subsidence_risk_higher_frequency() -> None:
    """Subsidence-risk properties should have higher claim frequency."""
    df = load_home(n_policies=20_000, seed=42)
    sub = df[df["is_subsidence_risk"]]
    no_sub = df[~df["is_subsidence_risk"]]
    freq_sub = sub["claim_count"].sum() / sub["exposure"].sum()
    freq_no_sub = no_sub["claim_count"].sum() / no_sub["exposure"].sum()
    assert freq_sub > freq_no_sub, (
        f"Subsidence frequency {freq_sub:.4f} should exceed no-subsidence {freq_no_sub:.4f}"
    )


def test_enhanced_security_lower_frequency_than_basic() -> None:
    """Enhanced security should reduce claim frequency vs. basic."""
    df = load_home(n_policies=20_000, seed=42)
    basic = df[df["security_level"] == "Basic"]
    enhanced = df[df["security_level"] == "Enhanced"]
    freq_basic = basic["claim_count"].sum() / basic["exposure"].sum()
    freq_enhanced = enhanced["claim_count"].sum() / enhanced["exposure"].sum()
    assert freq_enhanced < freq_basic, (
        f"Enhanced security frequency {freq_enhanced:.4f} should be below basic {freq_basic:.4f}"
    )


# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------


def test_home_params_exported() -> None:
    assert isinstance(HOME_TRUE_FREQ_PARAMS, dict)
    assert "intercept" in HOME_TRUE_FREQ_PARAMS
    assert "flood_zone_3" in HOME_TRUE_FREQ_PARAMS
    assert isinstance(HOME_TRUE_SEV_PARAMS, dict)
    assert "intercept" in HOME_TRUE_SEV_PARAMS
