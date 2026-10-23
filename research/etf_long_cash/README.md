# Core ETF long/cash research

- [Performance report](focused_payoff_context/README.md)
- [Executed statistical notebook](focused_payoff_context/payoff_context_review.ipynb)
- [Model and data pipeline](src/), [configurations](config/), [tests](tests/)

The source model remains v3.1: 15 separate ETF long/cash accounts, no portfolio
or options layer. Net Sharpe after costs is the primary measure. Evaluation ends
in 2023; **2024 onward remains reserved**.

## What is kept

The five model modules cover public data ingestion, expanding CART/RF/XGB
research, ETF-specific context, payoff experiments and the uncertainty utilities
required by the synthetic tests. Configurations preserve the original target,
timing, sizing and maturity assumptions.

The notebook keeps its full statistical tables and 61 embedded figures. Its
small prepared inputs are losslessly compressed CSV files, read directly by
pandas. Only columns used in the report are retained. The daily records are
display series, not a complete cash/share execution ledger.

Duplicate HTML/PNG exports, archived source copies, bulky audit traces, old
experiment reports, unused news modules and full training snapshots are removed
from the current version. The original complete study remains available locally.

## Run

Use an isolated Python 3.12 environment. From this folder:

    python -m pip install -r requirements-core.txt
    python -m unittest discover -s tests -v
    python scripts/validate_published_report.py

Open the notebook and Run All from its folder. It only reads the prepared
report inputs, and does not fetch market data or train a model. Rebuilding
statistics after a model change requires fresh prepared tables; cached source
statistics are not recomputed from the reduced display series.

Before new training, create a local reports/ directory. It is ignored by Git.
src/data_ingest.py prepares ignored local data/ histories;
src/research.py creates the baseline; src/etf_context.py prepares context; and
src/experiments_v3.py consumes the baseline forecasts, selections, cost ledgers
and metrics. The complete training inputs are not committed. Downloaded vendor
history may differ from the frozen study. Declare a new run version before
changing a completed experiment.

## Interpretation

Current results are exploratory. The parent study's 225 comparisons had no
Holm-adjusted result below 5%. New ETF/SPY intervals are post-hoc fixed-fit
descriptions. Cash earns zero, turnover costs are 5 bps one way, and adjusted
synthetic units approximate total return; XLF's 2016 distribution remains a
qualification. The original full-source audit passed 10,408 checks; current
validation checks only the retained display data and source-statistic consistency.

Virtual environments, caches, logs, raw downloads, fitted models and runtime
output are ignored. Cleanup uses a normal commit; older versions remain in Git
history.
