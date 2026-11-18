"""Small synthetic correctness tests; no synthetic production-data mode."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from suite import (Data, Strategy, backtest, ingest_data, permute_scores,
                   report, run, validate, volatility)


def fixture(days=110, assets=8):
    rng = np.random.default_rng(17)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, (days, assets)), axis=0))
    rows = np.arange(63, days - 1, 21)
    return Data(np.arange(days).astype("datetime64[D]"), tuple(f"S{i}" for i in range(assets)),
                close, np.ones((days, assets), bool), rows,
                rng.normal(size=(len(rows), assets)), np.zeros((len(rows), assets), dtype=int))


class SuiteTests(unittest.TestCase):
    def test_ingestion_is_blank_and_cannot_silently_load_demo_data(self):
        with self.assertRaisesRegex(NotImplementedError, "DATA INGEST"):
            ingest_data()

    def test_groups_and_score_multisets_are_preserved(self):
        data = fixture()
        data.groups[:, 4:] = 1
        mapping = np.full((25, 8), -1)
        scores, mapping = permute_scores(data, 0, mapping, 0, 42, 3)
        for columns in (np.arange(4), np.arange(4, 8)):
            for row in range(25):
                self.assertEqual(set(mapping[row, columns]), set(columns))
                np.testing.assert_array_equal(np.sort(scores[row, columns]), np.sort(data.scores[0, columns]))

    def test_draws_do_not_depend_on_batch_size(self):
        data = fixture()
        whole_map = np.full((19, 8), -1)
        left_map, right_map = whole_map[:7].copy(), whole_map[7:].copy()
        for query in range(len(data.decision_rows)):
            whole, whole_map = permute_scores(data, query, whole_map, 0, 42, 3)
            left, left_map = permute_scores(data, query, left_map, 0, 42, 3)
            right, right_map = permute_scores(data, query, right_map, 7, 42, 3)
            np.testing.assert_array_equal(whole, np.vstack([left, right]))
        a = run(data, variations=19, batch_size=7)
        b = run(data, variations=19, batch_size=19)
        np.testing.assert_array_equal(a.variation_equity, b.variation_equity)
        for key in a.variations:
            np.testing.assert_array_equal(a.variations[key], b.variations[key])

    def test_compatible_assignments_persist_and_future_data_cannot_change_them(self):
        data = fixture()
        initial = np.full((10, 8), -1)
        _, first = permute_scores(data, 0, initial, 0, 42, 3)
        _, second = permute_scores(data, 1, first, 0, 42, 3)
        np.testing.assert_array_equal(first, second)
        data.groups[2] = np.arange(8)
        _, unchanged = permute_scores(data, 0, initial, 0, 42, 3)
        np.testing.assert_array_equal(first, unchanged)

    def test_membership_changes_rematch_without_duplicate_donors(self):
        data = fixture()
        previous = np.full((10, 8), -1)
        _, previous = permute_scores(data, 0, previous, 0, 42, 3)
        data.eligible[data.decision_rows[1], 0] = False
        data.groups[1, 1:4] = 1
        _, mapping = permute_scores(data, 1, previous, 0, 42, 3)
        self.assertTrue((mapping[:, 0] == -1).all())
        for row in mapping:
            self.assertEqual(set(row[1:]), set(range(1, 8)))
            self.assertEqual(set(row[1:4]), {1, 2, 3})

    def test_identity_permutations_are_allowed(self):
        data = fixture(assets=2)
        _, mapping = permute_scores(data, 0, np.full((100, 2), -1), 0, 42, 3)
        self.assertTrue(np.any((mapping == [0, 1]).all(axis=1)))
        self.assertTrue(np.any((mapping == [1, 0]).all(axis=1)))

    def test_volatility_uses_only_the_observed_prefix(self):
        data = fixture()
        before = volatility(data.close, 63)
        data.close[90:] *= 4
        after = volatility(data.close, 63)
        np.testing.assert_array_equal(before[:90], after[:90])
        expected = np.std(data.close[1:64] / data.close[:63] - 1, axis=0, ddof=1)
        np.testing.assert_allclose(after[63], expected)

    def test_entry_and_final_liquidation_charge_real_turnover(self):
        data = fixture(days=70, assets=2)
        data.close[:] = 100
        rule = Strategy(cost_bps=2)
        path, turn, cost, _ = backtest(data, rule, volatility(data.close, 63), 1, randomized=False)
        fee = .0002
        expected = (1 - fee) / (1 + fee)
        self.assertAlmostEqual(path[-1, 0], expected, places=12)
        self.assertAlmostEqual(cost[0], fee / (1 + fee) + fee, places=12)
        self.assertGreater(turn[0], 1.99)

    def test_stops_execute_after_the_trigger_close_and_names_stay_blocked(self):
        data = fixture(days=75, assets=2)
        data.close[:] = 100
        data.close[66:, 0] = 90
        data.close[67:, 0] = 80
        data.close[68:, 0] = 70
        data.scores[:] = [1., -1.]
        rule = Strategy(cost_bps=0)
        path, _, _, stops = backtest(data, rule, volatility(data.close, 63), 1, randomized=False)
        # Enter at day64. Day66 observes the loss. Day67 still bears the
        # additional gap, then closes. Day68's further loss is avoided.
        np.testing.assert_allclose(path[[2, 3, 4], 0], [.95, .90, .90])
        self.assertEqual(stops[0], 1)

    def test_new_forecast_allows_reentry(self):
        data = fixture(days=90, assets=2)
        data.close[:] = 100
        data.close[66:, 0] = 90
        data.close[67:, 0] = 80
        data.close[86:, 0] = 72
        data.scores[:] = [1., -1.]
        path, _, _, _ = backtest(data, Strategy(cost_bps=0), volatility(data.close, 63), 1, randomized=False)
        self.assertLess(path[86 - 64, 0], path[85 - 64, 0])

    def test_missing_data_and_insufficient_matching_stop_without_fallbacks(self):
        data = fixture()
        data.close[70, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing/nonpositive"):
            run(data, variations=2)
        data = fixture()
        data.groups[:] = np.arange(8)
        with self.assertRaisesRegex(ValueError, "20%"):
            run(data, variations=2)

    def test_inactive_missing_quote_is_rejected_if_a_position_still_needs_it(self):
        data = fixture(days=75, assets=2)
        data.close[:] = 100
        data.scores[:] = [1., -1.]
        data.close[70, 0] = np.nan
        data.eligible[70, 0] = False
        with self.assertRaisesRegex(ValueError, "Missing held quote"):
            run(data, variations=2)

    def test_missing_entry_history_is_not_imputed(self):
        data = fixture(days=70, assets=2)
        data.close[2, 0] = np.nan
        data.eligible[2, 0] = False
        with self.assertRaisesRegex(ValueError, "Incomplete entry-volatility"):
            run(data, variations=2)

    def test_one_evaluation_session_is_not_enough_to_report_volatility(self):
        with self.assertRaisesRegex(ValueError, "two evaluation sessions"):
            run(fixture(days=65), variations=2)

    def test_report_contains_every_variation_and_only_two_output_files(self):
        data = fixture()
        result = run(data, variations=13, batch_size=5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report(result, data, Strategy(), root, seed=42, block_forecasts=3)
            self.assertEqual({p.name for p in root.iterdir()}, {"report.html", "variations.csv"})
            self.assertEqual(len((root / "variations.csv").read_text().splitlines()), 14)
            html = (root / "report.html").read_text()
            self.assertIn("<svg", html)
            self.assertIn("not future-return forecasts", html)


if __name__ == "__main__":
    unittest.main()
