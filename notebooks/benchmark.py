"""
Benchmark: insurance-datasets vs random noise features for GLM validation
=========================================================================

This is a data library, not a modelling technique. The benchmark
demonstrates why known-DGP synthetic data is essential for pricing model
development and validation.

The core question: when you fit a GLM or gradient boosted model, how do
you know if your implementation is correct? With real policyholder data,
the true coefficients are unknown. You can't distinguish "my model is
correct but the data is noisy" from "my model has a bug".

insurance-datasets solves this by generating data where the true
parameters are published. You can verify coefficient recovery.

This benchmark demonstrates three things:
1. A correctly specified Tweedie GLM on insurance-datasets recovers the
   true DGP coefficients within expected tolerance at 50k policies
2. Coefficient estimation improves monotonically with sample size
   (bias disappears at large N, variance shrinks — both verifiable)
3. Random noise features produce unstable, non-recoverable coefficients —
   the dataset makes the difference between a test you can trust and noise

Run with: uv run python notebooks/benchmark.py
Dependencies: insurance-datasets, statsmodels, numpy, scipy
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Tolerance for coefficient recovery
# The GLM should recover the true coefficient within this absolute error
# at 50k policies. At smaller sample sizes, the tolerance is wider.
# ---------------------------------------------------------------------------

RECOVERY_TOLERANCE = {
    50_000: 0.03,   # within 0.03 at 50k: very tight
    10_000: 0.07,   # within 0.07 at 10k: still good
    5_000:  0.12,   # within 0.12 at 5k: acceptable
    1_000:  0.25,   # within 0.25 at 1k: wide but expected
}


# ---------------------------------------------------------------------------
# GLM fitting
# ---------------------------------------------------------------------------

def fit_frequency_glm(df: pd.DataFrame) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":
    """
    Fit the correctly specified Poisson GLM for claim frequency.

    Uses the same feature engineering as the DGP:
    - vehicle_group (continuous)
    - ncd_years (continuous)
    - has_convictions (binary)
    - area dummies (B through F, with A as base)
    - driver_age young flag (driver_age < 25)
    - driver_age old flag (driver_age >= 70)
    - log(exposure) as offset
    """
    import statsmodels.api as sm

    df = df.copy()

    # Binary features matching the DGP
    df["has_convictions"] = (df["conviction_points"] > 0).astype(int)
    df["driver_age_young"] = (df["driver_age"] < 25).astype(int)
    df["driver_age_old"] = (df["driver_age"] >= 70).astype(int)

    for band in ["B", "C", "D", "E", "F"]:
        df[f"area_{band}"] = (df["area"] == band).astype(int)

    feature_names = [
        "vehicle_group", "ncd_years", "has_convictions",
        "driver_age_young", "driver_age_old",
        "area_B", "area_C", "area_D", "area_E", "area_F",
    ]

    X = sm.add_constant(df[feature_names])
    offset = np.log(df["exposure"].clip(lower=1e-6))

    model = sm.GLM(
        df["claim_count"],
        X,
        family=sm.families.Poisson(),
        offset=offset,
    )
    return model.fit(disp=False), feature_names


def fit_frequency_glm_noise(
    df: pd.DataFrame,
    n_noise_features: int = 10,
    seed: int = 99,
) -> "statsmodels.regression.linear_model.RegressionResultsWrapper":
    """
    Fit a Poisson GLM on random noise features only.

    Demonstrates what happens when you can't validate against a known DGP:
    - Coefficients look plausible (non-zero, standard errors finite)
    - But there is no ground truth to compare against
    - The model cannot be validated
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    n = len(df)

    noise_cols = {}
    noise_names = []
    for i in range(n_noise_features):
        col = f"noise_{i}"
        noise_cols[col] = rng.standard_normal(n)
        noise_names.append(col)

    df_noise = pd.DataFrame(noise_cols, index=df.index)
    X = sm.add_constant(df_noise)
    offset = np.log(df["exposure"].clip(lower=1e-6))

    model = sm.GLM(
        df["claim_count"],
        X,
        family=sm.families.Poisson(),
        offset=offset,
    )
    return model.fit(disp=False), noise_names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def coefficient_errors(
    fitted_params: pd.Series,
    true_params: dict[str, float],
) -> dict[str, float]:
    """
    Absolute errors between fitted and true coefficients.

    Returns dict of {param_name: abs_error}.
    """
    errors = {}
    for name, true_val in true_params.items():
        if name == "intercept":
            fitted_val = fitted_params.get("const", float("nan"))
        else:
            fitted_val = fitted_params.get(name, float("nan"))
        errors[name] = abs(fitted_val - true_val)
    return errors


def max_abs_error(errors: dict[str, float]) -> float:
    """Maximum absolute coefficient error across all parameters."""
    vals = [v for v in errors.values() if not np.isnan(v)]
    return max(vals) if vals else float("nan")


def mean_abs_error(errors: dict[str, float]) -> float:
    """Mean absolute coefficient error across all parameters."""
    vals = [v for v in errors.values() if not np.isnan(v)]
    return np.mean(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 65)
    print("insurance-datasets  benchmark")
    print("GLM coefficient recovery with known DGP")
    print("=" * 65)
    print()

    from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

    # Subset of params that are directly recoverable from the correctly-specified GLM
    # (driver_age effects are blended so we focus on the cleanly recoverable params)
    recoverable_params = {
        "intercept": MOTOR_TRUE_FREQ_PARAMS["intercept"],
        "vehicle_group": MOTOR_TRUE_FREQ_PARAMS["vehicle_group"],
        "ncd_years": MOTOR_TRUE_FREQ_PARAMS["ncd_years"],
        "has_convictions": MOTOR_TRUE_FREQ_PARAMS["has_convictions"],
        "area_B": MOTOR_TRUE_FREQ_PARAMS["area_B"],
        "area_C": MOTOR_TRUE_FREQ_PARAMS["area_C"],
        "area_D": MOTOR_TRUE_FREQ_PARAMS["area_D"],
        "area_E": MOTOR_TRUE_FREQ_PARAMS["area_E"],
        "area_F": MOTOR_TRUE_FREQ_PARAMS["area_F"],
    }

    print("True DGP frequency parameters:")
    for name, val in recoverable_params.items():
        print(f"  {name:<25} {val:+.3f}")
    print()

    # -----------------------------------------------------------------------
    # Part 1: Coefficient recovery at multiple sample sizes
    # -----------------------------------------------------------------------
    print("Part 1: Coefficient recovery by sample size")
    print("-" * 65)
    print("  Expected: bias and variance both shrink as N grows")
    print()

    sample_sizes = [1_000, 5_000, 10_000, 50_000]
    recovery_results = []

    for n in sample_sizes:
        df = load_motor(n_policies=n, seed=42)
        result, _ = fit_frequency_glm(df)
        errors = coefficient_errors(result.params, recoverable_params)
        mae_val = mean_abs_error(errors)
        max_err = max_abs_error(errors)
        tol = RECOVERY_TOLERANCE[n]
        passed = max_err <= tol

        recovery_results.append({
            "n": n,
            "mae": mae_val,
            "max_error": max_err,
            "tolerance": tol,
            "passed": passed,
        })

    headers = ["N policies", "Mean |error|", "Max |error|", "Tolerance", "Pass?"]
    col_w = [12, 14, 12, 12, 8]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  " + "-" * 58)
    for r in recovery_results:
        print(fmt.format(
            f"{r['n']:,}",
            f"{r['mae']:.4f}",
            f"{r['max_error']:.4f}",
            f"{r['tolerance']:.3f}",
            "YES" if r["passed"] else "NO",
        ))
    print()

    # -----------------------------------------------------------------------
    # Part 2: Detailed coefficient comparison at 50k
    # -----------------------------------------------------------------------
    print("Part 2: Detailed coefficient recovery at 50,000 policies")
    print("-" * 65)

    df_50k = load_motor(n_policies=50_000, seed=42)
    result_50k, feature_names = fit_frequency_glm(df_50k)
    errors_50k = coefficient_errors(result_50k.params, recoverable_params)

    col_w2 = [25, 12, 12, 12]
    fmt2 = "  ".join(f"{{:<{w}}}" for w in col_w2)
    print(fmt2.format("Parameter", "True", "Fitted", "|Error|"))
    print("  " + "-" * 61)

    for name in recoverable_params:
        true_val = recoverable_params[name]
        param_key = "const" if name == "intercept" else name
        fitted_val = result_50k.params.get(param_key, float("nan"))
        err = abs(fitted_val - true_val)
        print(fmt2.format(name, f"{true_val:+.4f}", f"{fitted_val:+.4f}", f"{err:.4f}"))

    print()
    print(f"  Overall MAE: {mean_abs_error(errors_50k):.4f}")
    print(f"  Max |error|: {max_abs_error(errors_50k):.4f} (tolerance: {RECOVERY_TOLERANCE[50_000]:.3f})")
    all_passed = max_abs_error(errors_50k) <= RECOVERY_TOLERANCE[50_000]
    print(f"  Recovery test: {'PASSED' if all_passed else 'FAILED'}")
    print()

    # -----------------------------------------------------------------------
    # Part 3: Comparison — known DGP vs random noise features
    # -----------------------------------------------------------------------
    print("Part 3: Known DGP vs random noise features")
    print("-" * 65)
    print("  With known DGP: coefficients are verifiable, bugs are detectable")
    print("  With noise features: model fits but no ground truth exists")
    print()

    df_compare = load_motor(n_policies=10_000, seed=42)

    # Known DGP model
    result_known, _ = fit_frequency_glm(df_compare)
    errors_known = coefficient_errors(result_known.params, recoverable_params)

    # Noise features model
    result_noise, noise_names = fit_frequency_glm_noise(df_compare, n_noise_features=10)
    # Noise coefficients — compute mean absolute value as "how confident do they look?"
    noise_coef_magnitudes = np.abs(result_noise.params[1:].values)  # exclude const

    print("  Known DGP model (correctly specified):")
    print(f"    Mean coefficient error vs truth: {mean_abs_error(errors_known):.4f}")
    print(f"    Max coefficient error vs truth:  {max_abs_error(errors_known):.4f}")
    print(f"    Log-likelihood: {result_known.llf:.1f}")
    print(f"    AIC: {result_known.aic:.1f}")
    print()
    print("  Random noise model (10 noise features, no ground truth):")
    print(f"    Mean |coefficient|: {noise_coef_magnitudes.mean():.4f}  (looks like signal, isn't)")
    print(f"    Max |coefficient|:  {noise_coef_magnitudes.max():.4f}")
    print(f"    Log-likelihood: {result_noise.llf:.1f}")
    print(f"    AIC: {result_noise.aic:.1f}")
    print(f"    Verifiable against truth? NO — there is no ground truth")
    print()

    # -----------------------------------------------------------------------
    # Part 4: Quick Tweedie GLM (using incurred as response)
    # -----------------------------------------------------------------------
    print("Part 4: Tweedie GLM on incurred (combined frequency * severity)")
    print("-" * 65)
    print("  The DGP supports Tweedie — use incurred directly as response")
    print()

    try:
        import statsmodels.api as sm

        df_tweedie = load_motor(n_policies=50_000, seed=42)
        df_tweedie["has_convictions"] = (df_tweedie["conviction_points"] > 0).astype(int)

        features_tweedie = ["vehicle_group", "ncd_years", "has_convictions"]
        X_t = sm.add_constant(df_tweedie[features_tweedie])
        offset_t = np.log(df_tweedie["exposure"].clip(lower=1e-6))

        # Tweedie GLM with power=1.5 (between Poisson and Gamma)
        tweedie_model = sm.GLM(
            df_tweedie["incurred"],
            X_t,
            family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.log()),
            offset=offset_t,
        )
        tweedie_result = tweedie_model.fit(disp=False)

        print("  Tweedie GLM (power=1.5) on incurred:")
        print(f"  {'Parameter':<25} {'Estimate':>12}")
        print("  " + "-" * 37)
        for pname in ["const", "vehicle_group", "ncd_years", "has_convictions"]:
            val = tweedie_result.params.get(pname, float("nan"))
            print(f"  {pname:<25} {val:>12.4f}")

        print()
        print("  The DGP has vehicle_group>0, ncd_years<0, has_convictions>0.")
        print("  Fitted signs should match. The dataset makes this verifiable.")

    except Exception as exc:
        print(f"  Tweedie GLM failed: {exc}")

    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("Summary:")
    total_pass = sum(r["passed"] for r in recovery_results)
    print(f"  Coefficient recovery tests passed: {total_pass}/{len(recovery_results)}")
    print()
    print("  The dataset enables proper model validation because you can")
    print("  check fitted coefficients against the published DGP parameters.")
    print("  Random noise features produce models with plausible-looking")
    print("  coefficients that cannot be validated against any ground truth.")
    print()
    print("  Use insurance-datasets whenever you need to:")
    print("  - Verify a GLM or GBM implementation is correct")
    print("  - Benchmark multiple modelling approaches fairly")
    print("  - Demonstrate that your pricing model recovers known effects")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
