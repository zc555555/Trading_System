"""Unit tests for the P1 evaluation layer: rank IC + purged walk-forward."""

import numpy as np
import pandas as pd

from evaluation.metrics import daily_rank_ic, summarize_ic, portfolio_metrics
from evaluation.factor_training import purged_inner_split
from evaluation.purged_walk_forward import WalkForwardConfig, generate_folds


def _panel(n_dates=10, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for d in dates:
        target = rng.normal(0, 1, n_names)
        rows.append(pd.DataFrame({
            "date": d,
            "symbol": [f"S{i}" for i in range(n_names)],
            "future_return": target,
        }))
    return pd.concat(rows, ignore_index=True)


class TestDailyRankIC:
    def test_perfect_ranking_gives_ic_one(self):
        df = _panel()
        df["pred"] = df["future_return"] * 3 + 7  # monotone transform
        ic = daily_rank_ic(df, "pred")
        assert np.allclose(ic, 1.0)

    def test_inverted_ranking_gives_minus_one(self):
        df = _panel()
        df["pred"] = -df["future_return"]
        ic = daily_rank_ic(df, "pred")
        assert np.allclose(ic, -1.0)

    def test_noise_is_near_zero_and_insignificant(self):
        df = _panel(n_dates=100, n_names=100, seed=1)
        rng = np.random.default_rng(2)
        df["pred"] = rng.normal(0, 1, len(df))
        s = summarize_ic(daily_rank_ic(df, "pred"))
        assert abs(s["ic_mean"]) < 0.05
        assert abs(s["t_stat"]) < 3

    def test_thin_dates_are_skipped(self):
        df = _panel(n_dates=5, n_names=10)  # below min_names_per_date=20
        df["pred"] = df["future_return"]
        ic = daily_rank_ic(df, "pred")
        assert len(ic) == 0


class TestPurgedSplits:
    def test_inner_split_has_purge_gap(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="B").values
        tr, va = purged_inner_split(dates, val_quantile=0.85, purge_days=6)
        # gap: exactly purge_days dates dropped between train end and val start
        all_sorted = np.sort(dates)
        gap = np.searchsorted(all_sorted, va[0]) - np.searchsorted(all_sorted, tr[-1]) - 1
        assert gap == 6
        assert tr[-1] < va[0]

    def test_walk_forward_folds_are_purged_and_non_overlapping(self):
        cfg = WalkForwardConfig(train_window=252, test_window=63, step=63,
                                horizon=1, embargo_days=5)
        dates = pd.date_range("2019-01-01", periods=800, freq="B").values
        folds = generate_folds(dates, cfg)
        assert len(folds) >= 2
        all_sorted = np.sort(dates)
        prev_test_end = None
        for train, test in folds:
            # purge gap between last train date and first test date
            gap = (np.searchsorted(all_sorted, test[0])
                   - np.searchsorted(all_sorted, train[-1]) - 1)
            assert gap == cfg.horizon + cfg.embargo_days
            # train shortened by exactly the purge
            assert len(train) == cfg.train_window - (cfg.horizon + cfg.embargo_days)
            assert len(test) == cfg.test_window
            # test windows non-overlapping and consecutive
            if prev_test_end is not None:
                assert test[0] > prev_test_end
            prev_test_end = test[-1]

    def test_no_train_label_reaches_test(self):
        """A label at train date t uses close(t + horizon); with the purge in
        place, t + horizon must still be strictly before the test start."""
        cfg = WalkForwardConfig(train_window=252, test_window=63, step=63,
                                horizon=1, embargo_days=5)
        dates = pd.date_range("2019-01-01", periods=500, freq="B").values
        all_sorted = np.sort(dates)
        for train, test in generate_folds(dates, cfg):
            last_label_day_idx = np.searchsorted(all_sorted, train[-1]) + cfg.horizon
            test_start_idx = np.searchsorted(all_sorted, test[0])
            assert last_label_day_idx < test_start_idx


class TestPortfolioMetrics:
    def test_constant_positive_returns(self):
        r = pd.Series(0.001, index=pd.date_range("2024-01-01", periods=252, freq="B"))
        m = portfolio_metrics(r)
        assert m["daily_win_rate"] == 1.0
        assert m["max_drawdown"] == 0.0
        assert m["total_return"] > 0.2

    def test_drawdown_sign(self):
        r = pd.Series([0.10, -0.20, 0.05])
        m = portfolio_metrics(r)
        assert -0.21 < m["max_drawdown"] < -0.19
