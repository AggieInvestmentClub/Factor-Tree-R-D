"""Financial and time-causality checks for the new ETF research implementation.

Run from the project directory with ``python -m unittest discover -s tests -v``.
Synthetic data make the expected cashflows and information cutoffs observable.
No provider requests, archived engines, or reserved research data are used.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import research


def price_frame(dates, opens, closes=None):
    dates = pd.DatetimeIndex(dates, name="Date")
    opens = np.asarray(opens, dtype=float)
    closes = opens if closes is None else np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "adj_open": opens, "adj_close": closes,
        "adj_high": np.maximum(opens, closes) * 1.01,
        "adj_low": np.minimum(opens, closes) * .99,
        "Volume": np.arange(len(dates), dtype=float) * 17 + 100_000,
    }, index=dates)


class FeatureCausalityTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2018-01-01", periods=430, name="Date")
        t = np.arange(len(self.dates))
        self.prices = {
            ticker: price_frame(self.dates, 80 + 10 * j + .03 * t + 3 * np.sin(t / (9 + j)),
                                80 + 10 * j + .03 * t + 3 * np.sin(t / (9 + j)) + .2 * np.cos(t / 5))
            for j, ticker in enumerate(["XLK", "SPY", "IEF", "GLD", "DBC"])
        }
        self.macro = pd.DataFrame({"macro_test": np.sin(t / 25) + 2}, index=self.dates)

    def test_features_match_when_future_data_do_not_exist(self):
        cutoff = self.dates[340]
        full = research.build_features(self.prices, "XLK", self.macro)
        prefix = research.build_features({k: v.loc[:cutoff] for k, v in self.prices.items()},
                                         "XLK", self.macro.loc[:cutoff])
        assert_frame_equal(full.loc[:cutoff], prefix)

    def test_future_price_peer_and_macro_shocks_do_not_change_past_features(self):
        cutoff = self.dates[340]
        before = research.build_features(self.prices, "XLK", self.macro)
        shocked = {k: v.copy() for k, v in self.prices.items()}
        for frame in shocked.values():
            frame.loc[frame.index > cutoff, ["adj_open", "adj_close", "adj_high", "adj_low"]] *= 7
            frame.loc[frame.index > cutoff, "Volume"] *= 100
        macro = self.macro.copy()
        macro.loc[macro.index > cutoff, "macro_test"] = -1e6
        after = research.build_features(shocked, "XLK", macro)
        assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])

    def test_not_yet_listed_peer_has_availability_indicator(self):
        later = {k: v.copy() for k, v in self.prices.items()}
        later["DBC"] = later["DBC"].iloc[300:]
        features = research.build_features(later, "XLK")
        self.assertTrue((features.loc[:self.dates[299], "rel_DBC_available_20"] == 0).all())
        self.assertTrue((features.loc[:self.dates[299], "rel_DBC_mom_20"] == 0).all())
        self.assertEqual(features.loc[self.dates[320], "rel_DBC_available_20"], 1)


class DecisionAndMaturityTests(unittest.TestCase):
    def setUp(self):
        # Monday 18 January is deliberately absent: execution must use Tuesday.
        self.dates = pd.bdate_range("2021-01-04", "2021-04-06").difference(
            pd.DatetimeIndex(["2021-01-18"]))
        self.price = price_frame(self.dates, 100 + np.arange(len(self.dates)))
        self.features = pd.DataFrame({"tech_test": np.arange(len(self.dates))}, index=self.dates)
        self.events = research.weekly_events(self.price, self.features)

    def test_weekly_decision_executes_at_next_actual_session(self):
        decision = pd.Timestamp("2021-01-15")
        self.assertEqual(self.events.loc[decision, "entry_date"], pd.Timestamp("2021-01-19"))
        self.assertNotEqual(self.events.loc[decision, "entry_date"], decision)
        self.assertEqual(self.events.loc[decision, "tech_test"], self.features.loc[decision, "tech_test"])

    def test_labels_use_next_open_to_horizon_open_with_declared_hurdle(self):
        decision = pd.Timestamp("2021-01-15")
        entry = self.dates.get_loc(pd.Timestamp("2021-01-19"))
        for horizon in (5, 10, 20):
            expected_end = self.dates[entry + horizon]
            expected_return = (self.price.adj_open.iloc[entry + horizon] /
                               self.price.adj_open.iloc[entry] - 1 -
                               research.CFG["label_roundtrip_hurdle"])
            self.assertEqual(self.events.loc[decision, f"label_end_{horizon}"], expected_end)
            self.assertAlmostEqual(self.events.loc[decision, f"y_{horizon}"], expected_return)

    def test_incomplete_final_week_is_not_a_decision(self):
        self.assertEqual(self.dates[-1], pd.Timestamp("2021-04-06"))
        self.assertEqual(self.events.index[-1], pd.Timestamp("2021-04-02"))
        self.assertTrue(pd.isna(self.events.iloc[-1].label_end_20))
        self.assertTrue(pd.isna(self.events.iloc[-1].y_20))

    def test_training_purges_labels_that_cross_or_touch_cutoff(self):
        cutoff = pd.Timestamp("2021-03-01")
        selected = research.mature_rows(self.events, cutoff, ["tech_test"])
        expected = self.events.loc[(self.events.index < cutoff) &
                                   (self.events.label_end_20 < cutoff)]
        self.assertGreater(len(selected), 0)
        self.assertLess(len(selected), (self.events.index < cutoff).sum())
        self.assertEqual(selected.index.tolist(), expected.index.tolist())
        self.assertTrue((selected.label_end_20 < cutoff).all())
        # The exact cutoff is intentionally excluded, even when its label exists.
        modified = self.events.copy()
        row = selected.index[-1]
        modified.loc[row, "label_end_20"] = cutoff
        self.assertNotIn(row, research.mature_rows(modified, cutoff, ["tech_test"]).index)


class SelfFinancingAccountTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2021-01-04", periods=4, name="Date")

    def test_all_cash_is_constant_and_cost_free(self):
        p = price_frame(self.dates, [100, 200, 50, 300])
        ledger = research.simulate(p, pd.Series(dtype=float), cost_bps=5)
        np.testing.assert_allclose(ledger.nav, 1)
        np.testing.assert_allclose(ledger.cash, 1)
        np.testing.assert_allclose(ledger[["shares", "costs", "turnover", "daily_return"]], 0)

    def test_shares_are_held_and_weight_drifts_without_a_new_decision(self):
        p = price_frame(self.dates, [100, 110, 80, 90], [110, 80, 90, 95])
        weights = pd.Series([.5], index=self.dates[:1])
        ledger = research.simulate(p, weights, cost_bps=0)
        np.testing.assert_allclose(ledger.nav, [1.05, .9, .95, .975])
        np.testing.assert_allclose(ledger.shares.iloc[:-1], .005)
        np.testing.assert_allclose(ledger.cash.iloc[:-1], .5)
        self.assertAlmostEqual(ledger.weight.iloc[0], .55 / 1.05)
        self.assertAlmostEqual(ledger.weight.iloc[1], .4 / .9)
        np.testing.assert_allclose(ledger.trade_value.iloc[1:], 0)
        np.testing.assert_allclose(ledger.turnover.iloc[1:-1], 0)
        self.assertEqual(ledger.shares.iloc[-1], 0)  # Declared final-close liquidation.

    def test_overnight_old_position_is_valued_before_open_rebalance(self):
        p = price_frame(self.dates, [100, 120, 160, 80], [100, 150, 160, 80])
        weights = pd.Series([1., 0.], index=self.dates[:2])
        ledger = research.simulate(p, weights, cost_bps=0)
        np.testing.assert_allclose(ledger.nav, [1., 1.2, 1.2, 1.2])
        self.assertAlmostEqual(ledger.opening_nav.iloc[1], 1.2)
        self.assertAlmostEqual(ledger.cash.iloc[1], 1.2)

    def test_roundtrip_cost_is_charged_on_actual_traded_notional(self):
        p = price_frame(self.dates, [100, 100, 100, 100])
        weights = pd.Series([1., 0.], index=self.dates[[0, 2]])
        c = 5 / 10000
        ledger = research.simulate(p, weights, cost_bps=5)
        self.assertAlmostEqual(ledger.nav.iloc[0], 1 / (1 + c))
        self.assertAlmostEqual(ledger.nav.iloc[-1], (1 - c) / (1 + c))
        self.assertAlmostEqual(ledger.costs.sum(), 2 * c / (1 + c))
        np.testing.assert_allclose(ledger.costs, ledger.trade_value.abs() * c)
        np.testing.assert_allclose(ledger.nav, ledger.cash + ledger.shares * p.adj_close)
        self.assertGreaterEqual(ledger.cash.min(), 0)

    def test_terminal_liquidation_cost_is_included(self):
        p = price_frame(self.dates, [100, 100, 100, 100])
        c = 10 / 10000
        ledger = research.simulate(p, pd.Series([1.], index=self.dates[:1]), cost_bps=10)
        self.assertAlmostEqual(ledger.nav.iloc[-1], (1 - c) / (1 + c))
        self.assertAlmostEqual(ledger.costs.iloc[-1], c / (1 + c))
        self.assertEqual(ledger.shares.iloc[-1], 0)
        self.assertAlmostEqual(ledger.cash.iloc[-1], ledger.nav.iloc[-1])

    def test_invalid_or_duplicate_weights_are_rejected(self):
        p = price_frame(self.dates, [100, 100, 100, 100])
        for invalid in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(weight=invalid), self.assertRaises(ValueError):
                research.simulate(p, pd.Series([invalid], index=self.dates[:1]))
        with self.assertRaises(ValueError):
            research.simulate(p, pd.Series([.2, .3], index=[self.dates[0], self.dates[0]]))


class ChronologicalReplayTests(unittest.TestCase):
    def test_december_decision_keeps_january_execution_in_inner_period(self):
        dates = pd.bdate_range("2014-01-01", "2022-12-30", name="Date")
        entry = pd.Timestamp("2022-01-03")
        decision = pd.Timestamp("2021-12-31")
        opens = np.where(dates > entry, 120., 100.)
        closes = np.where(dates >= entry, 120., 100.)
        price = price_frame(dates, opens, closes)
        features = pd.DataFrame({"tech_test": np.arange(len(dates)), "tech_vol_63": .15}, index=dates)
        events = research.weekly_events(price, features)
        self.assertEqual(events.loc[decision, "entry_date"], entry)

        def only_december_signal(train, test, columns, horizon, target, family):
            return np.asarray(test.index == decision, dtype=float), None

        with patch.object(research, "fit_predict", side_effect=only_december_signal):
            with patch.object(research, "simulate", wraps=research.simulate) as simulation:
                score, _ = research.eval_inner(events, price, 2023, "technical", 5,
                                                "classification", sizing="fixed")
        weights = simulation.call_args.args[1]
        self.assertIn(entry, weights.index)
        self.assertEqual(weights.loc[entry], 1.)
        self.assertGreater(score, 0)  # Captures the 20% entry-session gain.

    def test_inner_replay_carries_exposure_through_the_year_boundary(self):
        dates = pd.bdate_range("2014-01-01", "2020-12-31", name="Date")
        t = np.arange(len(dates))
        # A large year-opening gap makes omission of the boundary observable.
        levels = 100 * np.power(1.0001, t) * np.where(dates.year >= 2020, 1.2, 1.)
        price = price_frame(dates, levels)
        features = pd.DataFrame({"tech_test": t, "tech_vol_63": .15}, index=dates)
        events = research.weekly_events(price, features)

        def always_long(train, test, columns, horizon, target, family):
            return np.ones(len(test)), None

        with patch.object(research, "fit_predict", side_effect=always_long):
            actual_score, _ = research.eval_inner(events, price, 2021, "technical", 5,
                                                  "classification", sizing="fixed")
        eligible = events.loc[(events.index >= "2019-01-01") &
                              (events.index < "2021-01-01") &
                              (events.entry_date < pd.Timestamp("2021-01-01"))]
        weights = pd.Series(1., index=pd.DatetimeIndex(eligible.entry_date))
        expected = research.simulate(price, weights, start=weights.index[0],
                                     end=pd.Timestamp("2020-12-31"), cost_bps=5)
        expected_score = (expected.daily_return.mean() / expected.daily_return.std() * np.sqrt(252))
        self.assertAlmostEqual(actual_score, expected_score)
        self.assertGreater(expected.loc["2020-01-01", "daily_return"], .19)

    def test_reserved_prices_are_removed_before_feature_or_label_construction(self):
        dates = pd.bdate_range("2023-12-01", "2024-02-01", name="Date")
        prices = {"XLK": price_frame(dates, 100 + np.arange(len(dates))),
                  "SPY": price_frame(dates, 200 + np.arange(len(dates)))}

        class StopBeforeFitting(Exception):
            pass

        def inspect_inputs_then_stop(passed_prices, ticker, macro):
            self.assertEqual(ticker, "XLK")
            self.assertEqual(set(passed_prices), {"XLK", "SPY"})
            for frame in passed_prices.values():
                self.assertGreater(len(frame), 0)
                self.assertTrue((frame.index < pd.Timestamp(research.CFG["reserved_start"])).all())
            raise StopBeforeFitting

        with patch.object(research, "build_features", side_effect=inspect_inputs_then_stop):
            with self.assertRaises(StopBeforeFitting):
                research.run_instrument("XLK", prices, None, Path("unused-test-output"))


if __name__ == "__main__":
    unittest.main()
