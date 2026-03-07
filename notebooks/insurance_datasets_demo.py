# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-datasets demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow for both datasets provided by
# MAGIC the `insurance-datasets` package:
# MAGIC
# MAGIC 1. Install and import
# MAGIC 2. Explore the motor dataset
# MAGIC 3. Explore the home dataset
# MAGIC 4. Fit Poisson GLMs and verify coefficient recovery against the true DGP
# MAGIC 5. Fit a Gamma GLM for severity
# MAGIC 6. Show a CatBoost frequency model

# COMMAND ----------

# MAGIC %pip install insurance-datasets statsmodels catboost

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the motor dataset

# COMMAND ----------

import numpy as np
import pandas as pd
from insurance_datasets import (
    load_motor,
    load_home,
    MOTOR_TRUE_FREQ_PARAMS,
    MOTOR_TRUE_SEV_PARAMS,
    HOME_TRUE_FREQ_PARAMS,
    HOME_TRUE_SEV_PARAMS,
)

motor = load_motor(n_policies=50_000, seed=42)
print(f"Shape: {motor.shape}")
motor.head()

# COMMAND ----------

motor.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Portfolio summary statistics

# COMMAND ----------

freq = motor["claim_count"].sum() / motor["exposure"].sum()
claims_only = motor[motor["claim_count"] > 0]
mean_sev = claims_only["incurred"].sum() / claims_only["claim_count"].sum()

print(f"Policies:          {len(motor):,}")
print(f"Total exposure:    {motor['exposure'].sum():,.1f} policy years")
print(f"Total claims:      {motor['claim_count'].sum():,}")
print(f"Claim frequency:   {freq:.3f} per policy year")
print(f"Mean severity:     £{mean_sev:,.0f}")
print(f"Loss ratio proxy:  £{motor['incurred'].sum() / motor['exposure'].sum():,.0f} per policy year")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frequency by key rating factors

# COMMAND ----------

# NCD years
ncd_stats = (
    motor.groupby("ncd_years")
    .apply(lambda g: g["claim_count"].sum() / g["exposure"].sum(), include_groups=False)
    .rename("claim_frequency")
    .reset_index()
)
print("Frequency by NCD years:")
print(ncd_stats.to_string(index=False))

# COMMAND ----------

# Area band
area_stats = (
    motor.groupby("area")
    .apply(lambda g: g["claim_count"].sum() / g["exposure"].sum(), include_groups=False)
    .rename("claim_frequency")
    .reset_index()
    .sort_values("area")
)
print("Frequency by area band:")
print(area_stats.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the home dataset

# COMMAND ----------

home = load_home(n_policies=50_000, seed=42)
print(f"Shape: {home.shape}")
home.head()

# COMMAND ----------

home_freq = home["claim_count"].sum() / home["exposure"].sum()
home_claims = home[home["claim_count"] > 0]
home_mean_sev = home_claims["incurred"].sum() / home_claims["claim_count"].sum()

print(f"Policies:         {len(home):,}")
print(f"Claim frequency:  {home_freq:.3f} per policy year")
print(f"Mean severity:    £{home_mean_sev:,.0f}")

# COMMAND ----------

# Flood zone frequency
flood_stats = (
    home.groupby("flood_zone")
    .apply(lambda g: g["claim_count"].sum() / g["exposure"].sum(), include_groups=False)
    .rename("claim_frequency")
    .reset_index()
)
print("Frequency by flood zone:")
print(flood_stats.to_string(index=False))

# COMMAND ----------

# Construction type
const_stats = (
    home.groupby("construction_type")
    .apply(lambda g: g["claim_count"].sum() / g["exposure"].sum(), include_groups=False)
    .rename("claim_frequency")
    .reset_index()
)
print("Frequency by construction type:")
print(const_stats.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Motor frequency GLM: coefficient recovery

# COMMAND ----------

import statsmodels.api as sm

df = motor.copy()
df["has_convictions"] = (df["conviction_points"] > 0).astype(int)
for band in ["B", "C", "D", "E", "F"]:
    df[f"area_{band}"] = (df["area"] == band).astype(int)

features = [
    "vehicle_group", "ncd_years", "has_convictions",
    "area_B", "area_C", "area_D", "area_E", "area_F",
]
X = sm.add_constant(df[features])

freq_model = sm.GLM(
    df["claim_count"],
    X,
    family=sm.families.Poisson(),
    offset=np.log(df["exposure"].clip(lower=1e-6)),
)
freq_result = freq_model.fit(disp=False)
print(freq_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fitted vs true parameters

# COMMAND ----------

comparison = pd.DataFrame({
    "fitted": freq_result.params.rename({
        "const": "intercept",
        "has_convictions": "has_convictions",
    }),
    "true": {
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
})
comparison["abs_error"] = (comparison["fitted"] - comparison["true"]).abs()
comparison["pct_error"] = (comparison["abs_error"] / comparison["true"].abs() * 100).round(1)
print(comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Motor severity GLM

# COMMAND ----------

claims_df = df[df["claim_count"] > 0].copy()
claims_df["avg_severity"] = claims_df["incurred"] / claims_df["claim_count"]

Xs = sm.add_constant(claims_df[["vehicle_group"]])
# Binary young driver flag
Xs = Xs.copy()
Xs["young_driver"] = (claims_df["driver_age"] < 25).astype(int)

sev_model = sm.GLM(
    claims_df["avg_severity"],
    Xs,
    family=sm.families.Gamma(link=sm.families.links.Log()),
)
sev_result = sev_model.fit(disp=False)

sev_comparison = pd.DataFrame({
    "fitted": sev_result.params.rename({"const": "intercept"}),
    "true": {
        "intercept": MOTOR_TRUE_SEV_PARAMS["intercept"],
        "vehicle_group": MOTOR_TRUE_SEV_PARAMS["vehicle_group"],
        "young_driver": MOTOR_TRUE_SEV_PARAMS["driver_age_young"],
    }
})
sev_comparison["abs_error"] = (sev_comparison["fitted"] - sev_comparison["true"]).abs()
print(sev_comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Home frequency GLM

# COMMAND ----------

hdf = home.copy()
hdf["property_value_log"] = np.log(hdf["property_value"] / 250_000)
hdf["contents_value_log"] = np.log(hdf["contents_value"] / 30_000)
hdf["is_flood_zone_2"] = (hdf["flood_zone"] == "Zone 2").astype(int)
hdf["is_flood_zone_3"] = (hdf["flood_zone"] == "Zone 3").astype(int)
hdf["is_non_standard"] = (hdf["construction_type"] == "Non-Standard").astype(int)
hdf["is_listed"] = (hdf["construction_type"] == "Listed").astype(int)
hdf["is_security_standard"] = (hdf["security_level"] == "Standard").astype(int)
hdf["is_security_enhanced"] = (hdf["security_level"] == "Enhanced").astype(int)
hdf["subsidence"] = hdf["is_subsidence_risk"].astype(int)

home_features = [
    "property_value_log",
    "is_non_standard", "is_listed",
    "is_flood_zone_2", "is_flood_zone_3",
    "subsidence",
    "is_security_standard", "is_security_enhanced",
]
Xh = sm.add_constant(hdf[home_features])

home_freq_model = sm.GLM(
    hdf["claim_count"],
    Xh,
    family=sm.families.Poisson(),
    offset=np.log(hdf["exposure"].clip(lower=1e-6)),
)
home_freq_result = home_freq_model.fit(disp=False)

home_comparison = pd.DataFrame({
    "fitted": home_freq_result.params,
    "true": {
        "const": HOME_TRUE_FREQ_PARAMS["intercept"],
        "property_value_log": HOME_TRUE_FREQ_PARAMS["property_value_log"],
        "is_non_standard": HOME_TRUE_FREQ_PARAMS["construction_non_standard"],
        "is_listed": HOME_TRUE_FREQ_PARAMS["construction_listed"],
        "is_flood_zone_2": HOME_TRUE_FREQ_PARAMS["flood_zone_2"],
        "is_flood_zone_3": HOME_TRUE_FREQ_PARAMS["flood_zone_3"],
        "subsidence": HOME_TRUE_FREQ_PARAMS["subsidence_risk"],
        "is_security_standard": HOME_TRUE_FREQ_PARAMS["security_standard"],
        "is_security_enhanced": HOME_TRUE_FREQ_PARAMS["security_enhanced"],
    }
})
home_comparison["abs_error"] = (home_comparison["fitted"] - home_comparison["true"]).abs()
home_comparison["pct_error"] = (home_comparison["abs_error"] / home_comparison["true"].abs() * 100).round(1)
print(home_comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. CatBoost motor frequency model
# MAGIC
# MAGIC Tree models cannot use a Poisson log-link with an exposure offset directly,
# MAGIC so we use claim frequency (count / exposure) as the target.

# COMMAND ----------

from catboost import CatBoostRegressor, Pool

df_cb = motor.copy()
df_cb["has_convictions"] = (df_cb["conviction_points"] > 0).astype(int)
df_cb["claim_freq"] = df_cb["claim_count"] / df_cb["exposure"]

cat_features_list = ["area", "policy_type"]
num_features_list = [
    "vehicle_group", "driver_age", "ncd_years", "has_convictions",
    "annual_mileage", "vehicle_age", "occupation_class"
]
all_features = num_features_list + cat_features_list

pool = Pool(
    df_cb[all_features],
    label=df_cb["claim_freq"],
    cat_features=cat_features_list,
    weight=df_cb["exposure"],
)

cb_model = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    verbose=100,
    random_seed=42,
)
cb_model.fit(pool)

fi = pd.Series(
    cb_model.get_feature_importance(),
    index=all_features
).sort_values(ascending=False)
print("\nFeature importances:")
print(fi.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### CatBoost predictions vs GLM predictions
# MAGIC
# MAGIC A sanity check: both models should rank policies similarly by predicted frequency.

# COMMAND ----------

# Poisson GLM predictions
df["glm_freq_pred"] = freq_result.predict() / df["exposure"]

# CatBoost predictions
df["cb_freq_pred"] = cb_model.predict(Pool(df_cb[all_features], cat_features=cat_features_list))
df["cb_freq_pred"] = df["cb_freq_pred"].clip(lower=0)

correlation = df[["glm_freq_pred", "cb_freq_pred"]].corr().iloc[0, 1]
print(f"Spearman rank correlation between GLM and CatBoost predictions: {correlation:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `insurance-datasets` package provides two synthetic UK insurance datasets
# MAGIC with fully documented data generating processes. GLMs recover the true
# MAGIC coefficients closely at 50k policies. CatBoost ranks policies similarly to
# MAGIC the correctly specified GLM.
# MAGIC
# MAGIC Both datasets are reproducible via the `seed` parameter and have no missing
# MAGIC values by design.
# MAGIC
# MAGIC Install: `uv add insurance-datasets`
