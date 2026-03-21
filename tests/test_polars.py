"""
Tests for the polars=True output path in load_motor and load_home.

All tests use pytest.importorskip("polars") so they are automatically
skipped if polars is not installed. The polars output tests focus on
correctness of the conversion — the underlying data generation is already
covered by test_motor.py and test_home.py.
"""

from __future__ import annotations

import pytest

from insurance_datasets import load_motor, load_home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_polars_df(df: object, n_rows: int, n_cols: int) -> None:
    """Check that df is a polars DataFrame with the expected shape."""
    pl = pytest.importorskip("polars")
    assert isinstance(df, pl.DataFrame), f"Expected polars.DataFrame, got {type(df)}"
    assert df.shape == (n_rows, n_cols), f"Expected shape ({n_rows}, {n_cols}), got {df.shape}"


# ---------------------------------------------------------------------------
# load_motor polars tests
# ---------------------------------------------------------------------------


class TestLoadMotorPolars:
    def test_returns_polars_dataframe(self) -> None:
        pytest.importorskip("polars")
        df = load_motor(n_policies=100, seed=0, polars=True)
        _assert_polars_df(df, 100, 18)

    def test_column_names_match_pandas(self) -> None:
        pytest.importorskip("polars")
        import pandas as pd
        from insurance_datasets import load_motor

        df_pd = load_motor(n_policies=100, seed=0)
        df_pl = load_motor(n_policies=100, seed=0, polars=True)
        assert df_pl.columns == list(df_pd.columns)

    def test_row_count_correct(self) -> None:
        pytest.importorskip("polars")
        for n in [50, 99, 500]:
            df = load_motor(n_policies=n, seed=1, polars=True)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_data_values_match_pandas(self) -> None:
        """The numeric totals from polars and pandas outputs must agree."""
        pl = pytest.importorskip("polars")
        import pandas as pd

        df_pd = load_motor(n_policies=500, seed=42)
        df_pl = load_motor(n_policies=500, seed=42, polars=True)

        assert abs(df_pl["incurred"].sum() - df_pd["incurred"].sum()) < 1e-6
        assert abs(df_pl["exposure"].sum() - df_pd["exposure"].sum()) < 1e-6
        assert df_pl["claim_count"].sum() == df_pd["claim_count"].sum()

    def test_default_still_returns_pandas(self) -> None:
        pytest.importorskip("polars")  # ensure polars is available, proving the default is deliberate
        import pandas as pd

        df = load_motor(n_policies=100, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_policy_id_sequential(self) -> None:
        pytest.importorskip("polars")
        n = 200
        df = load_motor(n_policies=n, seed=0, polars=True)
        assert list(df["policy_id"].to_list()) == list(range(1, n + 1))

    def test_no_nulls(self) -> None:
        pytest.importorskip("polars")
        df = load_motor(n_policies=500, seed=0, polars=True)
        assert df.null_count().sum_horizontal()[0] == 0


# ---------------------------------------------------------------------------
# load_home polars tests
# ---------------------------------------------------------------------------


class TestLoadHomePolars:
    def test_returns_polars_dataframe(self) -> None:
        pytest.importorskip("polars")
        df = load_home(n_policies=100, seed=0, polars=True)
        _assert_polars_df(df, 100, 16)

    def test_column_names_match_pandas(self) -> None:
        pytest.importorskip("polars")
        import pandas as pd

        df_pd = load_home(n_policies=100, seed=0)
        df_pl = load_home(n_policies=100, seed=0, polars=True)
        assert df_pl.columns == list(df_pd.columns)

    def test_row_count_correct(self) -> None:
        pytest.importorskip("polars")
        for n in [50, 99, 500]:
            df = load_home(n_policies=n, seed=1, polars=True)
            assert len(df) == n, f"Expected {n} rows, got {len(df)}"

    def test_data_values_match_pandas(self) -> None:
        pl = pytest.importorskip("polars")
        import pandas as pd

        df_pd = load_home(n_policies=500, seed=42)
        df_pl = load_home(n_policies=500, seed=42, polars=True)

        assert abs(df_pl["incurred"].sum() - df_pd["incurred"].sum()) < 1e-6
        assert abs(df_pl["exposure"].sum() - df_pd["exposure"].sum()) < 1e-6
        assert df_pl["claim_count"].sum() == df_pd["claim_count"].sum()

    def test_default_still_returns_pandas(self) -> None:
        pytest.importorskip("polars")
        import pandas as pd

        df = load_home(n_policies=100, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_policy_id_sequential(self) -> None:
        pytest.importorskip("polars")
        n = 200
        df = load_home(n_policies=n, seed=0, polars=True)
        assert list(df["policy_id"].to_list()) == list(range(1, n + 1))

    def test_no_nulls(self) -> None:
        pytest.importorskip("polars")
        df = load_home(n_policies=500, seed=0, polars=True)
        assert df.null_count().sum_horizontal()[0] == 0


# ---------------------------------------------------------------------------
# ImportError when polars not installed
# ---------------------------------------------------------------------------


def test_motor_importerror_without_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    """If polars is not importable, passing polars=True must raise ImportError."""
    import insurance_datasets.motor as motor_mod

    monkeypatch.setattr(motor_mod, "_POLARS_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install polars"):
        load_motor(n_policies=10, seed=0, polars=True)


def test_home_importerror_without_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    """If polars is not importable, passing polars=True must raise ImportError."""
    import insurance_datasets.home as home_mod

    monkeypatch.setattr(home_mod, "_POLARS_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install polars"):
        load_home(n_policies=10, seed=0, polars=True)
