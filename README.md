# insurance-datasets
[![CI](https://github.com/burning-cost/insurance-datasets/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-datasets/actions/workflows/ci.yml)

Synthetic UK insurance datasets with known data generating processes. Built for testing pricing models.

When you are developing a GLM, a gradient boosted tree model, or any other pricing algorithm, you need data where you know what the right answer is. Real policyholder data is messy, access-controlled, and the true coefficients are unknown. This package gives you clean, realistic synthetic data where the true parameters are published — so you can verify your implementation produces the right coefficients.

Two datasets are available:

- **Motor**: UK personal lines motor insurance. 18 columns covering driver age, NCD, ABI vehicle group, area band, and more. Frequency and severity from a known Poisson-Gamma DGP.
- **Home**: UK household insurance. 16 columns covering property value, flood zone, construction type, subsidence risk, and security level. Same Poisson-Gamma structure.

## Installation

```bash
uv add insurance-datasets
```

Or with pip:

```bash
pip install insurance-datasets
```

Requires Python 3.10+. Dependencies: numpy, pandas.

## Quick start

```python
from insurance_datasets import load_motor, load_home

motor = load_motor(n_policies=50_000, seed=42)
home = load_home(n_policies=50_000, seed=42)

print(motor.head())
print(motor.dtypes)
```

### Motor dataset

```python
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS, MOTOR_TRUE_SEV_PARAMS

df = load_motor(n_policies=50_000, seed=42)

# One row per policy
# policy_id, inception_date, expiry_date, accident_year,
# vehicle_age, vehicle_group, driver_age, driver_experience,
# ncd_years, ncd_protected, conviction_points, annual_mileage,
# area, occupation_class, policy_type,
# claim_count, incurred, exposure

print(f"Claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.3f}")
print(f"Mean incurred (claims only): {df[df['claim_count']>0]['incurred'].mean():.0f}")
```

### Home dataset

```python
from insurance_datasets import load_home, HOME_TRUE_FREQ_PARAMS, HOME_TRUE_SEV_PARAMS

df = load_home(n_policies=50_000, seed=42)

# policy_id, inception_date, expiry_date, accident_year,
# region, property_value, contents_value, construction_type,
# flood_zone, is_subsidence_risk, security_level, bedrooms,
# property_age_band, claim_count, incurred, exposure

z1 = df[df["flood_zone"] == "Zone 1"]
z3 = df[df["flood_zone"] == "Zone 3"]
ratio = (z3["claim_count"].sum() / z3["exposure"].sum()) / (z1["claim_count"].sum() / z1["exposure"].sum())
print(f"Zone 3 vs Zone 1 frequency ratio: {ratio:.2f}x  (true DGP implies exp(0.85) = 2.34x)")
```

## Data generating processes

### Motor: true parameters

The frequency model is Poisson with a log-linear predictor:

```
log(lambda) = log(exposure) + intercept
            + vehicle_group_coef * vehicle_group_value
            + driver_age_young * I(driver_age < 25)
            + driver_age_old * I(driver_age >= 70)
            + ncd_years_coef * ncd_years_value
            + area_B * I(area == 'B')  ...  + area_F * I(area == 'F')
            + has_convictions * I(conviction_points > 0)
```

```python
from insurance_datasets import MOTOR_TRUE_FREQ_PARAMS, MOTOR_TRUE_SEV_PARAMS

print(MOTOR_TRUE_FREQ_PARAMS)
# {'intercept': -3.2, 'vehicle_group': 0.025, 'driver_age_young': 0.55,
#  'driver_age_old': 0.3, 'ncd_years': -0.12, 'area_B': 0.1, 'area_C': 0.2,
#  'area_D': 0.35, 'area_E': 0.5, 'area_F': 0.65, 'has_convictions': 0.45}

print(MOTOR_TRUE_SEV_PARAMS)
# {'intercept': 7.8, 'vehicle_group': 0.018, 'driver_age_young': 0.25}
```

The age effect blends linearly between 25 and 30 — ages 25-29 have a partial young-driver load. The severity model uses a Gamma distribution with shape=2 (coefficient of variation ~0.71).

### Home: true parameters

```python
from insurance_datasets import HOME_TRUE_FREQ_PARAMS, HOME_TRUE_SEV_PARAMS

print(HOME_TRUE_FREQ_PARAMS)
# {'intercept': -2.8, 'property_value_log': 0.18,
#  'construction_non_standard': 0.4, 'construction_listed': 0.25,
#  'flood_zone_2': 0.3, 'flood_zone_3': 0.85, 'subsidence_risk': 0.55,
#  'security_standard': -0.1, 'security_enhanced': -0.25}

print(HOME_TRUE_SEV_PARAMS)
# {'intercept': 8.1, 'property_value_log': 0.35,
#  'flood_zone_3': 0.45, 'contents_value_log': 0.22}
```

The home severity model uses Gamma with shape=1.5 (CV ~0.82). Household claims have higher severity volatility than motor — large flood and escape-of-water events create a fatter tail.

## GLM coefficient recovery example

The point of known DGP parameters is that you can verify your model implementation. Here is a worked example with `statsmodels`:

```python
import numpy as np
import statsmodels.api as sm
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)

# Create binary conviction flag (matches the DGP)
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)

# Area dummies (A is the base level)
for band in ["B", "C", "D", "E", "F"]:
    df[f"area_{band}"] = (df["area"] == band).astype(int)

features = [
    "vehicle_group", "ncd_years", "has_convictions",
    "area_B", "area_C", "area_D", "area_E", "area_F",
]
X = sm.add_constant(df[features])

model = sm.GLM(
    df["claim_count"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(df["exposure"].clip(lower=1e-6)),
)
result = model.fit(disp=False)

print("Parameter recovery:")
print(f"  vehicle_group: fitted={result.params['vehicle_group']:.4f}  true={MOTOR_TRUE_FREQ_PARAMS['vehicle_group']:.4f}")
print(f"  ncd_years:     fitted={result.params['ncd_years']:.4f}  true={MOTOR_TRUE_FREQ_PARAMS['ncd_years']:.4f}")
print(f"  convictions:   fitted={result.params['has_convictions']:.3f}  true={MOTOR_TRUE_FREQ_PARAMS['has_convictions']:.3f}")
```

At 50k policies, slope estimates should be within a few percent of the true values. The intercept will differ if you omit any factor from the true DGP — the model absorbs the omitted effects into the intercept.

## CatBoost example

If you are calibrating a gradient boosted tree and want to verify it respects the known monotonic relationships:

```python
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from insurance_datasets import load_motor

df = load_motor(n_policies=50_000, seed=42)
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)

features = ["vehicle_group", "driver_age", "ncd_years", "has_convictions", "annual_mileage"]
X = df[features]
y = df["claim_count"] / df["exposure"]

pool = Pool(X, label=y)
model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0)
model.fit(pool)

# Feature importances should rank ncd_years and driver_age highly
fi = pd.Series(model.get_feature_importance(), index=features).sort_values(ascending=False)
print(fi)
```

## Design choices

**Why Poisson-Gamma and not something more exotic?** GLMs are still the industry standard for personal lines pricing. The DGP matches what a correctly specified production model would use. If you want to test Tweedie models, use the raw `incurred` as the response — the data supports it.

**Why no missing values?** This is a testing dataset. Missing value imputation is a separate problem. Mixing the two makes it harder to isolate algorithm correctness.

**Why 50,000 policies as the default?** Below about 10,000 policies, coefficient estimates become noisy enough that a correct implementation can look wrong. At 50,000 the estimates are stable. For quick unit tests, 1,000-5,000 is sufficient.

**Why is the home DGP simpler than motor?** The motor DGP has more interacting factors because motor pricing in the UK has more rating variables and more data to calibrate them. The home DGP reflects a less mature pricing environment where the dominant factors (flood, construction, subsidence) are fewer and cleaner.

## Running the tests

Tests require `statsmodels` for the GLM recovery check:

```bash
uv add --dev statsmodels
uv run pytest
```

## Licence

MIT. See [LICENSE](LICENSE).
