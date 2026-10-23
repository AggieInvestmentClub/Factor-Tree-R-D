# Payoff + ETF context — v3.1

[Open or download the executed statistical notebook](payoff_context_review.ipynb).
It contains all 15 ETF sections, 139 executed code cells and 61 embedded figures.
Each equity chart shows payoff + context, ETF buy-and-hold and S&P 500 buy-and-hold.

The notebook includes performance/risk, drawdowns and recovery, trade expectancy,
holding periods, sizing/exposure, costs, annual/monthly stability, forecasting
errors, payoff calibration, factor usage and benchmark confidence intervals.

## Run locally

Install ../requirements-core.txt in Python 3.12, open this notebook from its
folder, and Run All. The three src/ helpers read the necessary data/*.csv.gz
tables directly. No downloads, model fitting or reserved-period scoring occur.
The notebook outputs carry the figures, so duplicate PNG and HTML exports are
not needed.

The retained data is a compact reporting snapshot: daily NAV/return/exposure
series, saved predictions and prepared statistics. It omits full execution
ledgers and raw OOF fitting observations. Original cashflow/OOF audits were run
on the complete local study; the concise verification record distinguishes
those preserved checks from validation of this lean report.

Sharpe is net of 5 bps one-way actual-turnover costs, with zero cash/risk-free
yield. Labels and holding periods differ; the detailed definitions are in the
notebook. All observations precede 2024. These results remain exploratory.

[Core model and setup guide](../README.md)
