# v39 cross-sectional portfolio backtest

Frozen benchmark and volatility-scaled stop, evaluated January 2, 2024 through
November 11, 2026 (719 sessions). This package contains the actual research
engine, its local source dependencies, all three annual portfolio checkpoints,
input hashes, reference returns, and reproducibility checks.

**The data check is qualified, not a claim of perfect historical data.** Both
forward accounts have 15 provisional corporate-action valuation sessions.
The source histories are reconstructed historical data, not an as-of-vintage
database. The forward period was previously examined in an older project and
is not a pristine holdout. See [DATA.md](DATA.md).

## Run the frozen forward backtest

Use Python 3.12 in a virtual environment. The recorded Windows environment is
in [environment.json](environment.json). Run these commands from this folder:

```text
python -m pip install -r requirements.txt
python v39.py verify-package
python v39.py check-data --input-root /path/to/v39_private_inputs --profile replay
python v39.py stage --input-root /path/to/v39_private_inputs --work-root /path/to/v39_run --profile replay
python v39.py run --work-root /path/to/v39_run --stage replay
python v39.py compare-reference --work-root /path/to/v39_run
```

Replace the example paths with your own; on Windows, quote paths with spaces.
Inputs and the work directory must be outside the repository. Use a new work
directory for each run. `stage` verifies every input's SHA-256 before any pickle
is loaded, then creates physical copies. It never changes the source data.

The repository intentionally excludes licensed vendor histories. Obtain the
separate, authorized private input pack from the project owner. Git alone is
enough to inspect the model and analyze saved strategy returns; it is not
enough to rebuild the stock-level backtest. Only load trusted pickle inputs.

## Rebuild features and refit the forward models

The larger `rebuild` input profile includes frozen pre-2024 training artifacts.
This rebuilds forward features, 5/10/21-session labels, the Huber efficacy gate,
risk arrays, annual 2025/26 models, targets, and both daily accounts:

```text
python v39.py stage --input-root /path/to/v39_private_inputs --work-root /path/to/v39_rebuild --profile rebuild --refit
python v39.py run --work-root /path/to/v39_rebuild --stage rebuild
python v39.py compare-reference --work-root /path/to/v39_rebuild
python verify_data.py --input-root /path/to/v39_private_inputs --work-root /path/to/v39_rebuild
```

The 2024 checkpoint is the sealed starting model, trained only on pre-2024
data. Its training source and complete parameter-selection record are included.
The supported rebuild preserves that checkpoint and the pre-2024 training
artifacts; it does **not** reconstruct the original vendor captures or refit
every historical model from 2007. The original `fit_2024.py` and legacy
acquisition/analysis scripts are provenance source, not standalone entry points
for this input pack. Use the launcher commands above.

Each stage runs in a separate process because the original source versions
have different `common.py` modules. The launcher restricts its own computation
to one CPU/BLAS thread. Outputs are under
`WORK_ROOT/current/v39_forward_validation/`; `reproduction.json` compares all
ledger columns and model tensors against the references.

## Contents

| Location | Purpose |
|---|---|
| [MODEL.md](MODEL.md) | Features, objective, training chronology, portfolio and stop rules |
| [DATA.md](DATA.md) | Universe, time alignment, schemas, exclusions and valuation qualifications |
| `engine/` | Frozen v39 source plus required older modules; path-only portability edits |
| `checkpoints/` | Exact 2024/25/26 portfolio weights and training metadata |
| `specification/` | Frozen design, feature list and upstream policy protocols |
| `data_contract/input_manifest.json` | Exact private inputs, sizes, SHA-256 and profiles |
| `reference/` | Forward account ledgers, linked historical return series and summary metrics |
| `verification/` | Original qualification, fresh input checks and reproduction results |
| `provenance/` | Original/distributed source hashes and package inventory |

The public design metadata has normalized paths. The private pack retains the
original sealed metadata so design and model hashes continue to agree.
Original research files were not edited. The model's math and trading rules
were not changed for this handoff.
The package `.gitattributes` preserves exact file bytes across Git checkouts
so Windows newline conversion does not invalidate the recorded hashes.

## Saved results and historical context

| Forward account | CAGR | Sharpe (zero risk-free) | Maximum drawdown |
|---|---:|---:|---:|
| Benchmark | 8.59% | 0.792 | -10.90% |
| Volatility stop | 7.17% | 0.708 | -10.29% |

Returns include 2 bps per side, with zero borrow fees, financing and cash yield.
These results retain the conditional valuations described above.
`reference/daily_returns_full.csv.gz` links v38 development returns (2007–2023)
to the v39 forward returns, with SPY for comparison. The accounts restart in
cash at the 2024 boundary; this is not an uninterrupted account simulation.
Full-history source issues are preserved in `verification/full_history_source_audit.json`.
SPY uses adjusted total-return wealth, buying at the first close and selling
at the last close of each reported period, with 2 bps at each end.

## Manual GitHub upload

Copy the complete **`v39_backtest`** folder to `research/v39_backtest` in
`AggieInvestmentClub/Factor-Tree-R-D`, preserving its subdirectories.
Commit that folder. Keep `v39_private_inputs` and all runtime work folders
outside the repository. The package `.gitignore` provides additional protection.
No new license is assigned to vendor data by this package.
