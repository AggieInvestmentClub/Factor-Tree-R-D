STATISTICS-HEAVY REPORTING EDITION

The source model and saved trades remain v3.1. This reporting extension
has139executed code cells and61embedded figures. Equity sections retain
only payoff+context, ETFbuy/hold andSPYbuy/hold curves.

Added diagnostics include all-account performance/trade dashboards;
annual/monthly returns and rollingSharpe; Sortino/Calmar/Ulcer/tailrisk;
drawdown duration/recovery; episode expectancy/concentration;
classification Brier/logloss/confusion counts; regression errors;
outer payoff reliability; earlier OOFcalibration support; exact weekly
policy-account intervals; context availability; factor/exposure stability;
cost stress; common/early-late stability; and paired Sharpe intervals.

Full machine-readable tables are indata/: monthly_returns.csv,
annual_performance.csv,risk_statistics.csv,drawdown_episodes.csv,
forecast_summary.csv,probability_reliability.csv,payoff_reliability.csv,
weekly_policy_intervals.csv,weekly_interval_summary.csv and the original
episode/factor/model-setting files. The61figures are also saved separately.
The notebook and HTML contain their plotted outputs for offline reading.

New ETF/SPY95%intervals are post-hoc descriptive fixed-fit block-bootstrap
intervals, not new confirmation tests. The prior225-comparison Holm history
is displayed separately. No additional strategy was fitted or promoted.
data/diagnostic_audit.json records10408independent statistics checks;
verification.json andmanifest.json verify the final export.

All return quantities in detailed CSVs are decimal fractions unless their
column explicitly identifies counts, ratios, calendar days or sessions.
Original source NAV/cost amounts remain normalized account currency.
Monthly/yearly measurements retain existing positions and fees. Forecast
labels can overlap, and weekly-policy returns differ from horizon labels.
Temporal tables are cached saved diagnostics; when input ledgers or metric
definitions change, regenerate the statistical tables before analysis.

PAYOFF + ETF CONTEXT — FOCUSED NOTEBOOK

Open payoff_context_review.ipynb. Saved outputs are already executed.
For GitHub, upload this entire focused_payoff_context folder, preserving
data/ and src/. To rerun, install requirements.txt in a Python 3.12
environment, choose that kernel, and Run All from this folder.

This notebook reports the latest local completed model run, v3.1.
There is no saved v3.6 run in the source workspace. The reporting export
does not assign a new model version. No fitting, fetching, trading or
reserved-period scoring occurs when notebook cells are rerun.

Only the combined payoff_context procedure appears as a trading model.
Every ETF has three curves: that procedure, its own buy-and-hold, and
SPY buy-and-hold as the S&P 500 proxy. Accounts start at NAV 1 immediately
before the first included open; first-session movement/cost is retained.
5 bps one-way turnover fees and final liquidation apply; cash earns zero.
Every account is separate. No portfolio allocation or option positions.

Wins/losses refer to completed continuous invested episodes; position
resizing stays inside an episode. Net P&L includes all actual fees.
Forecast horizon and realized holding duration are different measures.
Position statistics separate target sizing from drifted daily exposure.
Factor importance is in-sample split importance, not causal evidence.

All evaluated observations precede 2024. These historical results remain
exploratory; none of the parent study's 225 new comparisons passed its
Holm 5% adjustment. Vendor-adjusted synthetic units approximate total
return; XLF's 2016 in-kind distribution still needs action reconciliation.
Macro and FinBERT are not eligible fitted factors in this saved strategy.

reference_sources/ contains the training-source/config snapshots and
parent audit provenance. It is reference material; the full historical
training dataset and complete research environment remain in the parent
project. The portable notebook uses only the curated local result files.
