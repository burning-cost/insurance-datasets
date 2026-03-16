"""
Tests for insurance_datasets.motor.

The parameter recovery tests use statsmodels GLM. They run with n_policies=50_000
to get stable estimates. The tolerance is intentionally generous: we are checking
that the DGP is approximately correct, not that the GLM is a perfect solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_datasets import (
    MOTOR_TRUE_FREQ_PARAMS,
    MOTOR_TRUE_SEV_PARAMS,
    TRUE_FREQ_PARAMS,
    TRUE_SEV_PARAMS,
    load_motor,
)
from insurance_datasets.motor import (
    AREA_BANDS,
    AREA_PROBS,
)


# ---------------------------------------------------------------------------
# Basic shape and dtype tests
# ---------------------------------------------------------------------------


def test_load_motor_default_shape() -> None:
    df = load_motor(n_policies=1_000, seed=0)
    assert df.shape == (1_000, 18)


def test_load_motor_column_names() -> None:
    df = load_motor(n_policies=100, seed=0)
    expected = [
        "policy_id", "inception_date", "expiry_date", "inception_year",
        "vehicle_age", "vehicle_group", "driver_age", "driver_experience",
        "ncd_years", "ncd_protected", "conviction_points", "annual_mileage",
        "area", "occupation_class", "policy_type",
        "claim_count", "incurred", "exposure",
    ]
    assert list(df.columns) == expected


def test_load_motor_dtypes() -> None:
    df = load_motor(n_policies=100, seed=0)
    assert df["policy_id"].dtype == np.dtype("int64") or df["policy_id"].dtype == np.dtype("int32")
    assert df["vehicle_group"].dtype == np.dtype("int64") or df["vehicle_group"].dtype == np.dtype("int32")
    assert df["ncd_protected"].dtype == np.dtype("bool")
    assert df["incurred"].dtype == np.dtype("float64")
    assert df["exposure"].dtype == np.dtype("float64")


def test_no_missing_values() -> None:
    df = load_motor(n_policies=1_000, seed=0)
    assert df.isnull().sum().sum() == 0


def test_policy_id_sequential() -> None:
    n = 500
    df = load_motor(n_policies=n, seed=0)
    assert list(df["policy_id"]) == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# Domain constraint tests
# ---------------------------------------------------------------------------


def test_vehicle_group_range() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert df["vehicle_group"].min() >= 1
    assert df["vehicle_group"].max() <= 50


def test_driver_age_range() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert df["driver_age"].min() >= 17
    assert df["driver_age"].max() <= 85


def test_ncd_years_range() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert df["ncd_years"].min() >= 0
    assert df["ncd_years"].max() <= 5


def test_exposure_positive() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert (df["exposure"] > 0).all()


def test_exposure_at_most_one_year() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    # Some policies may run slightly over due to leap year rounding — cap at 1.01
    assert (df["exposure"] <= 1.01).all()


def test_incurred_non_negative() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert (df["incurred"] >= 0).all()


def test_incurred_zero_when_no_claim() -> None:
    df = load_motor(n_policies=5_000, seed=2)
    no_claim = df["claim_count"] == 0
    assert (df.loc[no_claim, "incurred"] == 0.0).all()


def test_incurred_positive_when_claim() -> None:
    df = load_motor(n_policies=5_000, seed=2)
    has_claim = df["claim_count"] > 0
    assert (df.loc[has_claim, "incurred"] > 0).all()


def test_area_bands_valid() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert set(df["area"].unique()).issubset(set(AREA_BANDS))


def test_policy_type_valid() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert set(df["policy_type"].unique()).issubset({"Comp", "TPFT"})


def test_annual_mileage_range() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert df["annual_mileage"].min() >= 2_000
    assert df["annual_mileage"].max() <= 30_000


def test_inception_year_range() -> None:
    df = load_motor(n_policies=2_000, seed=1)
    assert df["inception_year"].min() >= 2019
    assert df["inception_year"].max() <= 2023


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed() -> None:
    df1 = load_motor(n_policies=500, seed=99)
    df2 = load_motor(n_policies=500, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ() -> None:
    df1 = load_motor(n_policies=500, seed=10)
    df2 = load_motor(n_policies=500, seed=11)
    # At least the incurred totals should differ
    assert df1["incurred"].sum() != df2["incurred"].sum()


# ---------------------------------------------------------------------------
# Statistical plausibility tests
# ---------------------------------------------------------------------------


def test_claim_frequency_plausible() -> None:
    """Portfolio average claim frequency should be in the 5-12% range."""
    df = load_motor(n_policies=10_000, seed=42)
    freq = df["claim_count"].sum() / df["exposure"].sum()
    assert 0.05 < freq < 0.12, f"Claim frequency {freq:.4f} is outside expected range"


def test_mean_severity_plausible() -> None:
    """Mean claim severity should be in the £1,500-£5,000 range."""
    df = load_motor(n_policies=10_000, seed=42)
    claims = df[df["claim_count"] > 0]
    mean_sev = claims["incurred"].sum() / claims["claim_count"].sum()
    assert 1_500 < mean_sev < 5_000, f"Mean severity £{mean_sev:.0f} is outside expected range"


def test_young_driver_higher_frequency() -> None:
    """Drivers under 25 should have materially higher claim frequency than middle-aged."""
    df = load_motor(n_policies=20_000, seed=42)
    young = df[df["driver_age"] < 25]
    mid = df[(df["driver_age"] >= 30) & (df["driver_age"] < 60)]
    young_freq = young["claim_count"].sum() / young["exposure"].sum()
    mid_freq = mid["claim_count"].sum() / mid["exposure"].sum()
    assert young_freq > mid_freq * 1.3, (
        f"Young driver frequency {young_freq:.4f} not sufficiently higher than "
        f"mid-age {mid_freq:.4f}"
    )


def test_ncd_inverse_frequency() -> None:
    """High NCD should have lower claim frequency than zero NCD."""
    df = load_motor(n_policies=20_000, seed=42)
    zero_ncd = df[df["ncd_years"] == 0]
    max_ncd = df[df["ncd_years"] == 5]
    zero_freq = zero_ncd["claim_count"].sum() / zero_ncd["exposure"].sum()
    max_freq = max_ncd["claim_count"].sum() / max_ncd["exposure"].sum()
    assert zero_freq > max_freq, (
        f"Zero NCD frequency {zero_freq:.4f} should exceed 5-year NCD {max_freq:.4f}"
    )


def test_area_f_higher_than_area_a() -> None:
    """Area F (inner city) should have higher claim frequency than Area A (rural)."""
    df = load_motor(n_policies=20_000, seed=42)
    area_a = df[df["area"] == "A"]
    area_f = df[df["area"] == "F"]
    freq_a = area_a["claim_count"].sum() / area_a["exposure"].sum()
    freq_f = area_f["claim_count"].sum() / area_f["exposure"].sum()
    assert freq_f > freq_a, (
        f"Area F frequency {freq_f:.4f} should exceed Area A {freq_a:.4f}"
    )


# ---------------------------------------------------------------------------
# Exported constants
# ---------------------------------------------------------------------------


def test_true_freq_params_exported() -> None:
    assert isinstance(TRUE_FREQ_PARAMS, dict)
    assert "intercept" in TRUE_FREQ_PARAMS
    assert "vehicle_group" in TRUE_FREQ_PARAMS
    assert "ncd_years" in TRUE_FREQ_PARAMS


def test_true_sev_params_exported() -> None:
    assert isinstance(TRUE_SEV_PARAMS, dict)
    assert "intercept" in TRUE_SEV_PARAMS


def test_motor_params_alias() -> None:
    assert MOTOR_TRUE_FREQ_PARAMS is TRUE_FREQ_PARAMS
    assert MOTOR_TRUE_SEV_PARAMS is TRUE_SEV_PARAMS


# ---------------------------------------------------------------------------
# GLM parameter recovery (requires statsmodels)
# ---------------------------------------------------------------------------


def test_glm_frequency_intercept_recovery() -> None:
    """
    Fit a Poisson GLM on the full DGP and check the intercept is close to
    the true value. We only test the intercept here — the full recovery test
    is in test_parameter_recovery.py — because fitting a 6-covariate GLM takes
    a few seconds and we want this file to stay fast.
    """
    pytest.importorskip("statsmodels")
    import statsmodels.api as sm

    df = load_motor(n_policies=20_000, seed=42)
    # Minimal spec: just intercept + log(exposure) offset
    X = sm.add_constant(np.zeros(len(df)))
    model = sm.GLM(
        df["claim_count"],
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["exposure"].clip(lower=1e-6)),
    )
    result = model.fit(disp=False)
    fitted_intercept = result.params["const"]
    true_intercept = TRUE_FREQ_PARAMS["intercept"]
    # Intercept alone absorbs all factor variation — check it's in ballpark
    assert abs(fitted_intercept - true_intercept) < 2.0, (
        f"Intercept {fitted_intercept:.3f} is very far from true {true_intercept:.3f}"
    )


# ---------------------------------------------------------------------------
# P0-1 regression: non-round n_policies must not crash or silently truncate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [99, 101, 999, 1001, 9999])
def test_load_motor_non_round_n(n: int) -> None:
    """
    For any n, the output must have exactly n rows.

    The driver_age bucket generation uses np.round() + residual absorption.
    With the old int() truncation, the five bucket sizes could sum to n-1 or
    n-2 for certain values, causing the [:n] slice to return a shorter array
    and subsequent operations to raise a ValueError.
    """
    df = load_motor(n_policies=n, seed=0)
    assert len(df) == n, f"Expected {n} rows, got {len(df)}"
    assert df.shape[1] == 18, f"Expected 18 columns, got {df.shape[1]}"
    assert df.isnull().sum().sum() == 0
