"""Independent checks for new experimental calibration and audit machinery."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import audit_v3
import experiments_v3 as experiment
import research
from pandas.testing import assert_frame_equal


def prices(dates):
    t = np.arange(len(dates))
    close = 100 * np.exp(.0002 * t + .01 * np.sin(t / 9))
    opening = close * (1 + .002 * np.cos(t / 5))
    return pd.DataFrame({"adj_open": opening, "adj_close": close,
                         "adj_high": np.maximum(opening, close) * 1.01,
                         "adj_low": np.minimum(opening, close) * .99,
                         "Volume": 100000 + t}, index=pd.DatetimeIndex(dates, name="Date"))


class PayoffCalibrationTests(unittest.TestCase):
    def test_bin_payoff_shrinks_to_prior_oof_mean_and_empty_bin_uses_prior(self):
        frame = pd.DataFrame({"prediction": [.1] * 26 + [.9] * 26,
                              "outcome": [-.02] * 26 + [.03] * 26})
        fitted = experiment.fit_payoff_calibrator(frame, "classification")
        result = experiment.apply_calibrator(fitted, [.1, .9, .5])
        self.assertTrue(fitted["eligible"])
        np.testing.assert_allclose(result["expected_payoff"], [-.0075, .0175, .005])
        np.testing.assert_array_equal(result["bin_n"], [26, 26, 0])
        self.assertTrue(np.isfinite(result["dispersion"]).all())

    def test_insufficient_matured_oof_has_explicit_fallback(self):
        fitted = experiment.fit_payoff_calibrator(pd.DataFrame({"prediction": [.8] * 51,
                                                                 "outcome": [.01] * 51}), "classification")
        result = experiment.apply_calibrator(fitted, [.8, .2])
        self.assertFalse(fitted["eligible"])
        self.assertEqual(fitted["n"], 51)
        self.assertTrue(np.isnan(result["expected_payoff"]).all())
        np.testing.assert_array_equal(result["bin_n"], 0)

    def test_regression_bins_are_frozen_before_future_predictions(self):
        fitted = experiment.fit_payoff_calibrator(pd.DataFrame({"prediction": np.arange(100),
                                                                 "outcome": np.arange(100) / 1000}), "regression")
        snapshot = json.dumps(fitted, sort_keys=True)
        experiment.apply_calibrator(fitted, [-1e9, 1e9])
        self.assertEqual(snapshot, json.dumps(fitted, sort_keys=True))
        np.testing.assert_allclose(fitted["edges"], [19.8, 39.6, 59.4, 79.2])

    def test_constant_regression_predictions_remain_finite(self):
        fitted = experiment.fit_payoff_calibrator(pd.DataFrame({"prediction": np.ones(100),
                                                                 "outcome": np.linspace(-.03, .02, 100)}), "regression")
        result = experiment.apply_calibrator(fitted, [-1, 1, 2])
        self.assertTrue(np.isfinite(result["expected_payoff"]).all())
        self.assertTrue(np.isfinite(result["dispersion"]).all())

    def test_oof_fits_and_outcomes_obey_their_distinct_cutoffs(self):
        dates = pd.date_range("2010-01-01", "2022-12-30", freq="W-FRI")
        events = pd.DataFrame({"tech_test": np.arange(len(dates)),
                               "entry_date": dates + pd.offsets.BDay(1)}, index=dates)
        for horizon in (5, 10, 20):
            events[f"label_end_{horizon}"] = events.entry_date + pd.offsets.BDay(horizon)
            events[f"y_{horizon}"] = .01
        cutoff = pd.Timestamp("2022-01-01")
        events.loc[events.label_end_20 >= cutoff, "y_5"] = 999.

        def prior_fit(train, test, columns, target, label):
            fit_cutoff = pd.Timestamp(test.index[0].year, 1, 1)
            self.assertGreaterEqual(len(train), 104)
            self.assertTrue((train.label_end_20 < fit_cutoff).all())
            return np.repeat(.75, len(test)), None

        with patch.object(experiment, "fit_predict", side_effect=prior_fit):
            oof = experiment.prior_oof(events, ["tech_test"], 5, "classification", 2022, False)
        self.assertGreater(len(oof), 52)
        self.assertTrue((oof.label_end20 < cutoff).all())
        self.assertTrue((oof.label_end < cutoff).all())
        self.assertTrue((oof.fit_cutoff <= oof.decision_date).all())
        self.assertFalse((oof.outcome == 999).any())


class ExperimentClockAndAccountTests(unittest.TestCase):
    def test_explicit_EWMA_risk_estimate_precedes_generic_vol_suffix(self):
        frame = pd.DataFrame({"vol63": [.20, .40, .06], "ewma_vol": [.40, .20, .20],
                              "base_sizing": ["fixed", "fixed", "fixed"]})
        np.testing.assert_allclose(experiment.sizing_scale("ewma_vol", frame), [.25, .5, .5])
        np.testing.assert_allclose(experiment.sizing_scale("trend_10m_vol", frame), [.5, .25, 1.])
        np.testing.assert_allclose(experiment.sizing_scale("baseline", frame), 1.)

    def test_hysteresis_retains_only_prior_state_at_boundaries(self):
        actual = experiment.hysteresis([1, 1.1, 1, -1, -1.1, .5, 1.2], 1, -1)
        np.testing.assert_array_equal(actual, [False, True, True, True, False, False, True])

    def test_fast_account_matches_original_for_drifting_exposure_and_fees(self):
        dates = pd.bdate_range("2020-01-02", periods=100)
        price = prices(dates)
        weights = pd.Series([0., .8, 1., .3, 0., .5], index=dates[[0, 6, 17, 43, 72, 87]])
        for bps in (0, 5, 10):
            with self.subTest(cost_bps=bps):
                fast = experiment.fast_simulate(price, weights, cost_bps=bps)
                original = research.simulate(price, weights, cost_bps=bps)
                assert_frame_equal(fast, original, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_weekly_target_uses_next_actual_weekly_execution_open(self):
        dates = pd.bdate_range("2020-01-01", "2021-03-31").difference(pd.DatetimeIndex(["2021-01-18"]))
        price = prices(dates)
        context = pd.DataFrame(index=price.index)
        with patch.object(experiment, "load_context", return_value=context):
            events = experiment.prepare_events({"XLK": price, "SPY": price}, "XLK")
        row = events.loc[pd.Timestamp("2021-01-15")]
        self.assertEqual(row.entry_date, pd.Timestamp("2021-01-19"))
        self.assertEqual(row.weekly_label_end, pd.Timestamp("2021-01-25"))
        expected = price.loc["2021-01-25", "adj_open"] / price.loc["2021-01-19", "adj_open"] - 1 - .001
        self.assertAlmostEqual(row.y_weekly, expected)

    def test_completed_month_trend_feature_obeys_prefix_causality(self):
        dates = pd.bdate_range("2016-01-01", "2020-09-30")
        price = prices(dates)
        cutoff = pd.Timestamp("2020-07-17")
        with patch.object(experiment, "load_context", return_value=pd.DataFrame(index=price.index)):
            full = experiment.prepare_events({"XLK": price, "SPY": price}, "XLK")
        prefix_price = price.loc[:cutoff]
        with patch.object(experiment, "load_context", return_value=pd.DataFrame(index=prefix_price.index)):
            prefix = experiment.prepare_events({"XLK": prefix_price, "SPY": prefix_price}, "XLK")
        assert_frame_equal(full.loc[prefix.index, ["diagnostic_trend10m", "diagnostic_ewma_vol"]],
                           prefix[["diagnostic_trend10m", "diagnostic_ewma_vol"]])

    def test_loader_removes_reserve_before_feature_operations(self):
        dates = pd.bdate_range("2023-12-01", "2024-02-01")
        with patch.object(experiment.base, "load_prices", return_value={"XLK": prices(dates)}):
            result = experiment.pre_reserve_prices()
        self.assertTrue((result["XLK"].index < pd.Timestamp("2024-01-01")).all())


class PairedMultiplicityTests(unittest.TestCase):
    def test_block_moments_match_naive_resampling_including_short_last_block(self):
        rng = np.random.default_rng(19)
        for sessions in [17, 57, 60]:
            values = rng.normal(.001, .015, (4, sessions))
            starts = rng.integers(0, max(1, sessions - 20 + 1), size=(9, int(np.ceil(sessions / 20))))
            means, variances = audit_v3.sampled_block_moments(values, starts, block=20)
            for i, row in enumerate(starts):
                indices = np.concatenate([np.arange(start, min(start + 20, sessions)) for start in row])[:sessions]
                sample = values[:, indices]
                np.testing.assert_allclose(means[:, i], sample.mean(axis=1), rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(variances[:, i], sample.var(axis=1, ddof=1), rtol=1e-12, atol=1e-14)

    def test_zero_risk_free_sharpe_scaling_invariance_is_paired(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(.0005, .01, 300)
        frame = pd.DataFrame({"baseline": returns, "identical": returns,
                              "half_exposure": returns / 2},
                             index=pd.bdate_range("2010-01-01", periods=300))
        rows = audit_v3.joint_paired_bootstrap(frame, replications=50)
        for row in rows:
            self.assertAlmostEqual(row["delta_sharpe"], 0.)
            self.assertAlmostEqual(row["delta_sharpe_95pct_low"], 0.)
            self.assertAlmostEqual(row["delta_sharpe_95pct_high"], 0.)
            self.assertAlmostEqual(row["p_centered_bootstrap_one_sided_approx"], 1.)

    def test_holm_adjustment_is_monotone_and_preserves_input_order(self):
        actual = audit_v3.holm_adjust([.01, .04, .03])
        np.testing.assert_allclose(actual, [.03, .06, .06])
        np.testing.assert_allclose(audit_v3.holm_adjust([0., 1., .5]), [0., 1., 1.])

    def test_missing_sessions_cannot_silently_drop_from_bootstrap(self):
        with self.assertRaises(ValueError):
            audit_v3.joint_paired_bootstrap(pd.DataFrame({"baseline": [0., .01], "arm": [0., np.nan]}))


if __name__ == "__main__":
    unittest.main()
