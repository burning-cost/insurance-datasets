# insurance-datasets

[![CI](https://github.com/burning-cost/insurance-datasets/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-datasets/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-datasets)](https://pypi.org/project/insurance-datasets/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Synthetic UK insurance datasets with known data generating processes. Built for testing pricing models — this is the dataset used throughout the [Burning Cost training course](https://burning-cost.github.io/course/).

When you are developing a GLM, a gradient boosted tree, or any other pricing algorithm, you need data where you know what the right answer is. Real policyholder data is messy, access-controlled, and the true coefficients are unknown. This package gives you clean, realistic synthetic data where the true parameters are published — so you can verify your implementation produces the right coefficients.

Two datasets are available:

- **Motor**: UK personal lines motor insurance. 18 columns covering driver age, NCD, ABI vehicle group, area band, and more. Frequency and severity from a known Poisson-Gamma DGP.
- **Home**: UK household insurance. 16 columns covering property value, flood zone, construction type, subsidence risk, and security level. Same Poisson-Gamma structure.

## Installation

```bash
pip install insurance-datasets
```

Or with `uv`:

```bash
uv add insurance-datasets
```

Requires Python 3.10+. Dependencies: numpy, pandas.

## Quick start

```python
from insurance_datasets import load_motor, load_home

motor = load_motor(n_policies=50_000, seed=42)
home  = load_home(n_policies=50_000, seed=42)

print(motor.shape)   # (50000, 18)
print(home.shape)    # (50000, 16)
```

---

## Motor dataset

`load_motor()` returns one row per policy. Default: 50,000 policies, inception years 2019–2023.

### Columns

| Column | Type | Description |
|---|---|---|
| `policy_id` | int | Sequential identifier |
| `inception_date` | date | Policy start |
| `expiry_date` | date | Policy end (may be < 12 months for cancellations) |
| `inception_year` | int | Calendar year of inception — use for cohort splits |
| `vehicle_age` | int | 0–20 years |
| `vehicle_group` | int | ABI group 1–50 |
| `driver_age` | int | 17–85 |
| `driver_experience` | int | Years licensed |
| `ncd_years` | int | 0–5 (UK NCD scale) |
| `ncd_protected` | bool | Protected NCD flag |
| `conviction_points` | int | Total endorsement points (0 = clean) |
| `annual_mileage` | int | 2,000–30,000 miles |
| `area` | str | ABI area band A–F (A = rural/low risk, F = inner city) |
| `occupation_class` | int | 1–5 |
| `policy_type` | str | `'Comp'` or `'TPFT'` |
| `claim_count` | int | Number of claims in period |
| `incurred` | float | Total incurred cost (£); 0.0 if no claims |
| `exposure` | float | Earned years (< 1.0 for cancellations) |

### True DGP — frequency

Poisson frequency model with log-linear predictor:

```
log(lambda) = log(exposure) + intercept
            + vehicle_group_coef * vehicle_group
            + driver_age_young * I(driver_age < 25)
            + driver_age_old   * I(driver_age >= 70)
            + ncd_years_coef   * ncd_years
            + area_B * I(area == 'B') ... area_F * I(area == 'F')
            + has_convictions  * I(conviction_points > 0)
```

Ages 25–29 blend linearly from the young-driver load down to zero by age 30.

```python
from insurance_datasets import MOTOR_TRUE_FREQ_PARAMS

print(MOTOR_TRUE_FREQ_PARAMS)
# {'intercept': -3.2, 'vehicle_group': 0.025, 'driver_age_young': 0.55,
#  'driver_age_old': 0.3, 'ncd_years': -0.12, 'area_B': 0.1, 'area_C': 0.2,
#  'area_D': 0.35, 'area_E': 0.5, 'area_F': 0.65, 'has_convictions': 0.45}
```

The intercept of -3.2 is the baseline log-frequency for a reference policy (area A, no convictions, zero NCD, average vehicle group). `exp(-3.2) ≈ 4.1%` per year. The portfolio average frequency is higher because most policies sit above the reference level on at least one factor.

### True DGP — severity

Gamma severity model with shape=2 (coefficient of variation ~0.71):

```python
from insurance_datasets import MOTOR_TRUE_SEV_PARAMS

print(MOTOR_TRUE_SEV_PARAMS)
# {'intercept': 7.8, 'vehicle_group': 0.018, 'driver_age_young': 0.25}
```

Baseline severity at intercept 7.8 gives a mean of roughly £2,440.

---

## Home dataset

`load_home()` returns one row per policy. Default: 50,000 policies, inception years 2019–2023.

### Columns

| Column | Type | Description |
|---|---|---|
| `policy_id` | int | Sequential identifier |
| `inception_date` | date | Policy start |
| `expiry_date` | date | Policy end |
| `inception_year` | int | Calendar year of inception |
| `region` | str | UK region (ONS groupings, 12 values) |
| `property_value` | int | Buildings sum insured (£); regional log-normal |
| `contents_value` | int | Contents sum insured (£) |
| `construction_type` | str | `'Standard'`, `'Non-Standard'`, or `'Listed'` |
| `flood_zone` | str | Environment Agency zone: `'Zone 1'`, `'Zone 2'`, `'Zone 3'` |
| `is_subsidence_risk` | bool | High-subsidence area (London clay, Midlands) |
| `security_level` | str | `'Basic'`, `'Standard'`, or `'Enhanced'` |
| `bedrooms` | int | 1–5 |
| `property_age_band` | str | Construction era: Pre-1900, 1900–1945, 1945–1980, 1980–2000, Post-2000 |
| `claim_count` | int | Number of claims in period |
| `incurred` | float | Total incurred cost (£); 0.0 if no claims |
| `exposure` | float | Earned years |

### True DGP — frequency

```python
from insurance_datasets import HOME_TRUE_FREQ_PARAMS

print(HOME_TRUE_FREQ_PARAMS)
# {'intercept': -2.8, 'property_value_log': 0.18,
#  'construction_non_standard': 0.4, 'construction_listed': 0.25,
#  'flood_zone_2': 0.3, 'flood_zone_3': 0.85, 'subsidence_risk': 0.55,
#  'security_standard': -0.1, 'security_enhanced': -0.25}
```

`property_value_log` and `contents_value_log` are scaled as `log(value / reference)` — `log(property_value / 250_000)` and `log(contents_value / 30_000)` respectively.

### True DGP — severity

Gamma severity model with shape=1.5 (CV ~0.82 — household claims are more volatile than motor):

```python
from insurance_datasets import HOME_TRUE_SEV_PARAMS

print(HOME_TRUE_SEV_PARAMS)
# {'intercept': 8.1, 'property_value_log': 0.35,
#  'flood_zone_3': 0.45, 'contents_value_log': 0.22}
```

Baseline severity at intercept 8.1 gives a mean of roughly £3,300.

---

## Verifying your model against the true parameters

The point of a known DGP is that you can check your implementation. Here is a worked GLM example. Note that `driver_age` must be included — omitting it biases the NCD and convictions coefficients because the young-driver frequency load is partially absorbed into correlated factors.

```python
import numpy as np
import statsmodels.api as sm
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)
df["young_driver"] = (df["driver_age"] < 25).astype(int)
df["old_driver"] = (df["driver_age"] >= 70).astype(int)

for band in ["B", "C", "D", "E", "F"]:
    df[f"area_{band}"] = (df["area"] == band).astype(int)

features = [
    "vehicle_group", "young_driver", "old_driver",
    "ncd_years", "has_convictions",
    "area_B", "area_C", "area_D", "area_E", "area_F",
]
X = sm.add_constant(df[features])

result = sm.GLM(
    df["claim_count"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(df["exposure"].clip(lower=1e-6)),
).fit(disp=False)

print("Parameter recovery:")
print(f"  vehicle_group: fitted={result.params['vehicle_group']:.4f}  true={MOTOR_TRUE_FREQ_PARAMS['vehicle_group']:.4f}")
print(f"  ncd_years:     fitted={result.params['ncd_years']:.4f}  true={MOTOR_TRUE_FREQ_PARAMS['ncd_years']:.4f}")
print(f"  young_driver:  fitted={result.params['young_driver']:.3f}  true={MOTOR_TRUE_FREQ_PARAMS['driver_age_young']:.3f}")
print(f"  convictions:   fitted={result.params['has_convictions']:.3f}  true={MOTOR_TRUE_FREQ_PARAMS['has_convictions']:.3f}")
```

At 50k policies, slope estimates should be within a few percent of the true values. Omitting `young_driver` from the formula introduces omitted variable bias — NCD and convictions will both appear inflated because they are correlated with driver age in this portfolio.

### Verifying a flood zone relativity (home)

```python
from insurance_datasets import load_home

df = load_home(n_policies=50_000, seed=42)

z1 = df[df["flood_zone"] == "Zone 1"]
z3 = df[df["flood_zone"] == "Zone 3"]
ratio = (z3["claim_count"].sum() / z3["exposure"].sum()) / (z1["claim_count"].sum() / z1["exposure"].sum())
print(f"Zone 3 vs Zone 1 frequency ratio: {ratio:.2f}x")
# True DGP implies exp(0.85) = 2.34x — you should be close to this unadjusted
```

---

## Design choices

**Why Poisson-Gamma and not something more exotic?** GLMs are the industry standard for personal lines pricing in the UK. The DGP matches what a correctly specified production model would use. If you want to test Tweedie models, use raw `incurred` as the response — the data supports it.

**Why no missing values?** This is a testing dataset. Missing value imputation is a separate problem. Mixing the two makes it harder to isolate algorithm correctness.

**Why 50,000 policies as the default?** Below about 10,000 policies, coefficient estimates become noisy enough that a correct implementation can look wrong. At 50,000 the estimates are stable. For quick unit tests, 1,000–5,000 is sufficient.

**Why is the home DGP simpler than motor?** Motor pricing in the UK has more rating variables with stronger interactions. The home DGP reflects a less mature pricing environment where a handful of factors (flood zone, construction type, subsidence) dominate.

**Why `inception_year` and not `accident_year`?** In actuarial triangles, accident year means the year a loss *occurred* — which for a synthetic portfolio without claim development would need to be modelled separately from inception. This column is simply the year the policy incepted, so `inception_year` is the right term.

---

## Running the tests

Tests include a GLM coefficient recovery check and require `statsmodels`:

```bash
uv add --dev statsmodels
uv run pytest
```

---

## Capabilities

The notebook at `notebooks/insurance_datasets_demo.py` loads both datasets at 50,000 policies and runs Poisson GLMs for frequency and Gamma GLMs for severity on both motor and home, comparing fitted coefficients to the published true values. It demonstrates:

- **GLM coefficient recovery**: Motor frequency Poisson GLM recovers all major parameters (vehicle group, NCD, area band, convictions, driver age) within a few percent of their true values at 50k policies.
- **Severity recovery**: Gamma GLM with log link recovers the vehicle group and young driver severity parameters accurately.
- **Home DGP validation**: Flood zone, construction type, and subsidence coefficients are recovered from the home dataset, including the Zone 3 frequency uplift of roughly 2.3x relative to Zone 1.
- **Ground truth as a testing tool**: The `MOTOR_TRUE_FREQ_PARAMS` and `HOME_TRUE_FREQ_PARAMS` dicts let you quantify how far any modelling implementation deviates from the correctly specified answer — something impossible with real data.
- **Reproducibility**: All datasets are fully deterministic given a seed; `load_motor(seed=42)` always returns the same 50,000 policies.

---

## Related libraries

| Library | Why it's relevant |
|---|---|
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Generate portfolio-fitted synthetic data — use when you need data matched to your own book rather than a fixed DGP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | GLM interaction detection — use this dataset to validate that the CANN pipeline recovers known interaction structure |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation — this dataset gives a controlled environment to benchmark CV strategies |
| [insurance-validation](https://github.com/burning-cost/insurance-validation) | Model validation tools — use with this dataset to check validation metrics against known true parameters |

[All Burning Cost libraries and course](https://burning-cost.github.io/course/) →

---

## Licence

MIT. See [LICENSE](LICENSE).
