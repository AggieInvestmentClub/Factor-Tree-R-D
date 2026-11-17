# Significance tests for v3.1: work in progress (Manas)

*Saved Nov 17, 2026, mid-run. This folder is a work in progress and doesn't touch anything else
in the repo. When the tests are finished, move the final write-up into `docs/`.*

## What we're testing and why

v3.1 (15 ETF long/cash trees) earns positive returns with lower drawdowns, but the short-side
diagnosis and the volatility check (`docs/short_side_diagnosis.md`) suggested that the returns
come from **ETF drift plus being in cash about 30% of the time**, not from the factors picking
good weeks. These three tests check that and produce the "statistical validation" Joflay asked for.

| # | Test | What it randomizes | Null question |
|---|---|---|---|
| 1 | Random timing | Which weeks are cash (same count, run lengths and trades per ETF) | Does v3.1's timing beat coin-flip timing at the same exposure? |
| 2 | Label permutation | The training labels, circularly shifted vs the features | Does training on real labels beat training on noise? |
| 3 | Synthetic markets | The prices: block bootstrap with drift and volatility but no predictability | Does the pipeline "look good" even on markets with nothing to find? |

Common setup: weekly decisions 2008–2023, held from entry open to the next entry open,
5 bps one-way cost, 0/1 sizing (timing only), equal-capital portfolio across ETFs.
Annualized Sharpe = weekly mean/std × √52. **2024+ is untouched**; prices were downloaded
only through 2023-12-29.

## Status

| Test | Status | Result so far |
|---|---|---|
| #1 random timing | **Done**, 10,000 shuffles | v3.1 portfolio Sharpe 0.567 vs random median 0.622 (90% band 0.516–0.744). **20th percentile, p = 0.80.** 0 of 15 ETFs beat 95% of random; XLK at the 2nd percentile. |
| Real-data replica | **Done** | Portfolio Sharpe 0.479 vs always-long 0.602; cash ~35% of weeks. |
| #2 label permutation | **Paused at 90 of 200 runs** | Null median 0.495 (90% band 0.377–0.617). Real replica at the **41st percentile, p ≈ 0.59** (interim). 2 of 15 ETFs above the null's 95th percentile (IEF; about what chance gives). |
| #3 synthetic markets | **Paused at 2 of 200 runs** | Too few runs to read. |

Interim reading: **no evidence that the factors carry timing information.** Training on
scrambled labels does about as well as training on the real ones. Wait for the full #2 and #3
runs before quoting #2/#3 numbers.

## The replica (`common.py`)

Tests #2 and #3 retrain the trees many times, so `common.py` re-implements the core v2/v3.1
training loop from `research/etf_long_cash/src/research.py`:
- For each ETF and outer year: the same inner selection (horizon 5/10/20 × classification/
  regression × technical/relationships × fixed/vol sizing, scored on the two prior years).
- Then an expanding-window CART refit on matured rows, predicting that year.
- Differences from published v3.1: no ETF-context features, no payoff calibrator, and inner
  scores use the weekly evaluator.
- Check: on XLK it picks the same horizon/target as v3.1 in 14 of 16 years, and its weekly
  signals match 74% of the time.

Test #1 uses the **published** v3.1 signals (`forecast_outcomes.csv.gz`), not the replica.

## How to resume on another machine

1. `git pull` this repo.
2. **Price data** (gitignored, not in the repo). Either:
   - copy `research/etf_long_cash/data/` from the laptop (best: the same data as the saved
     runs), or
   - re-download through 2023 only:
     `research/etf_long_cash/.venv/Scripts/python research/etf_long_cash/src/data_ingest.py --cutoff 2023-12-29 --prices-only`
     run from inside `research/etf_long_cash/`. Then compare against `price_data_sha256.txt`
     by running `sha256sum *.csv` in `research/etf_long_cash/data/normalized/prices/`.
     **If any hash differs, Yahoo has revised its data: delete `results/*_runs.jsonl` and
     rerun everything** (real, permutation, synthetic, and test #1) so all runs use one dataset.
3. Python 3.12 environment (from the repo root):
   - `py -3.12 -m venv research/etf_long_cash/.venv`
   - `research/etf_long_cash/.venv/Scripts/python -m pip install -r research/etf_long_cash/requirements-core.txt`
4. Resume. The script skips seeds already saved in `results/*_runs.jsonl`. `--workers`
   defaults to the CPU count minus 2; each run takes ~2–3 min per worker.
   - `research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/retrain_tests.py --test permutation --n 200`
   - `research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/retrain_tests.py --test synthetic --n 200`
5. Summarize: `research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/summarize_retrain_tests.py`
   (writes `results/retrain_summary.json`; the current copy is an interim one).
6. Test #1 only needs rerunning if the price data changed:
   `research/etf_long_cash/.venv/Scripts/python wip/manas_significance_tests/random_timing_test.py`

## Files

| File | Purpose |
|---|---|
| `common.py` | Weekly evaluator plus the replica of the training loop |
| `random_timing_test.py` | Test #1 |
| `retrain_tests.py` | Real replica, #2 and #3; appends one JSON line per run |
| `summarize_retrain_tests.py` | p-values and percentiles for #2 and #3 |
| `price_data_sha256.txt` | Checksums of the price files used so far |
| `results/` | Saved outputs: `*_runs.jsonl`, test #1 tables, logs |

## Caveats to state in the write-up

- **Test #2 shifts labels; it doesn't scramble them row by row.** A circular shift keeps the
  labels' autocorrelation and overlap, so the null isn't too easy.
- **Inner model selection is redone inside every null run.** The noise models get the same
  chance to overfit as the real one, which is fair.
- **Test #3's bootstrap keeps patterns shorter than about a month** (mean block of 20
  sessions), so short-term momentum isn't fully removed.
- The ETFs share market moves, so per-ETF p-values aren't independent. Lead with the
  portfolio p-value.
- These are gross of borrow and financing; cash earns zero.

## Next steps after the tests

1. Finish #2 and #3, write up all three for Joflay (portfolio p-values first).
2. Then the cash-week direction-signal split test (trend, HYG/IEF, VIX term structure,
   downside vol, panic). It's described in `CLAUDE.md`.
3. Short model only if a factor separates cleanly; see "Mirrored short model" in `CLAUDE.md`.
