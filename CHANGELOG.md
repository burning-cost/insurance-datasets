# Changelog

## v0.1.3 (2026-03-22) [unreleased]
- fix: add Python 3.10/3.11 classifiers, missing URLs, consolidate dev deps
- fix: use plain string license field for universal setuptools compatibility
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.3 (2026-03-21)
- docs: replace pip install with uv add in README
- fix: add pyarrow to polars optional dep (fixes CI)
- Add polars=True output option to load_motor and load_home (v0.1.3)
- Add blog post link and community CTA to README
- Add Performance section with benchmark results from Databricks
- Add benchmark: GLM parameter recovery against known DGP for motor and home datasets
- QA batch 10: fix intercept comment, GLM example OVB, duplicate README section
- fix: P0/P1 bugs — rounding crash, OVB, wrong terminology, severity GLM (v0.1.2)
- Add uv.lock to pin verified dependency versions
- Pin statsmodels>=0.14.5 to fix scipy _lazywhere removal
- Add Related Libraries section to README
- Add capability demo and README Capabilities section
- Improve README: schema tables, course link, baseline numbers

## v0.1.0 (2026-03-09)
- polish README: add PyPI badge, related libraries cross-refs
- Add GitHub Actions CI workflow and test badge
- feat: initial release of insurance-datasets v0.1.0

