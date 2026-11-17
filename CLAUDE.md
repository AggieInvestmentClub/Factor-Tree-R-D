# Factor-Tree-R-D: context for Claude Code

I'm Manas, a member of the AIC Quant Fund "Trees" team, with Bryce (lead), Abhinav and Joflay (fund lead).
Repo: https://github.com/AggieInvestmentClub/Factor-Tree-R-D

## Deadline
Joflay wants every team to have **a working backtest with statistical validation**, sent to him
a reasonable time before the next Thursday meeting so he can review it.
Bryce asked me to figure out **why the model fails on the short side**.

## Project history (short)
- Based on arXiv 2405.13959, "Decision Trees for Intuitive Intraday Trading Strategies": a
  per-stock depth-4 CART on 9 technical indicators, with a 1/0 buy/out target. That paper has
  no costs, a single train/test split and inflated Sharpe annualization, and it gives no support
  for shorting.
- **v3.1** (`research/etf_long_cash/`): 15 separate ETF accounts, each with weekly long/cash
  decisions from CART/RF/XGB trees on about 40 technical and relationship factors, refit
  annually on an expanding window. 5 bps costs. 2024 onward is held out.
  Universe: XLK XLF XLE XLV XLI XLY XLP XLU XLB IEF TLT TIP GLD DBC VNQ.
  Weekly equal-capital portfolio 2012–2023: Sharpe 0.72 vs SPY 0.85. Lower drawdown than
  buy-and-hold, lower return.
- **v4.0** (`research/etf_long _short_flat/`): the same trees plus shorting. Sharpe fell in
  14 of 15 ETFs; short win rates were mostly 17–40%; 0 of 52 comparisons significant.
- **v40** (`BACKTEST RESULTS.pdf`, Nov 12): a new cross-sectional model that ranks stocks with
  annual PyTorch "log-growth" models, predicts 21 sessions ahead, and has a volatility-stop
  variant. Forward 2024–26: CAGR 8.6%, Sharpe 0.79 vs SPY 19.4% / 1.21. SPY beta about 0.5,
  alpha about 0, IC about 0.
- **v39 package** (`research/v39_backtest/`, pushed by Bryce Nov 13): the frozen engine source,
  2024/25/26 checkpoints, MODEL.md, DATA.md and reference returns. Forward 2024-01-02 to
  2026-11-11: benchmark CAGR 8.59%, Sharpe 0.792, max DD −10.9%; vol stop 7.17% / 0.708 /
  −10.3%. Rerunning the stock-level backtest needs Bryce's private input pack (licensed data,
  not in git). Its README says the forward period is not a pristine holdout.
- Abhinav's concept (`docs/Abhinav's Implementation Concept`) is long / put / exclude with
  deep-ITM 1–2 week puts. Not backtested. His `Code-Drafts/` script targets individual stocks
  and uses yfinance.

## My finding: why the short side fails
Files: `docs/short_side_diagnosis.md` and `scripts/short_side_diagnosis.py`. The script uses
`research/etf_long_cash/focused_payoff_context/data/forecast_outcomes.csv.gz`.
- After "cash" decisions, ETFs averaged **+16.6 bp/wk** forward (43.5% fell), versus +14.7
  after "long" decisions. Cash weeks weren't followed by declines.
- Forward return after cash calls was positive in 14 of 15 ETFs (XLK +71 bp/wk). DBC was
  the only negative one, and it's the only ETF with negative overall drift.
- The rank correlation between expected payoff and forward return is slightly negative in
  12 of 15 ETFs (median −0.03).
- Cash weeks are **more volatile in both directions**. The model is a **risk filter, not a
  direction predictor**: "cash" means uncertain, not down. Shorting those weeks bets against
  upward drift and takes on rebound risk.

## Volatility check (`scripts/cash_week_volatility_check.py`)
- Return per risk (mean/std of forward weekly return) is the same for long and cash weeks
  (0.072 vs 0.074). Cash weeks are more volatile in 11/15 ETFs, but they are not worse weeks.
- The model's ratio beat always-long in only 3/15 ETFs (median 0.061 vs 0.082). v3.1's lower
  drawdown comes from lower exposure, not timing. That's an honest caveat for Joflay.

## Significance tests: IN PROGRESS, pick up here
Folder: `wip/manas_significance_tests/`. **Read its `HANDOFF.md` first.** It has the status,
results and resume steps.
- **#1 random timing (done):** v3.1's real decisions vs 10,000 random-timing versions with
  identical exposure, run lengths and trade count. Portfolio Sharpe 0.567 vs random median
  0.622. **20th percentile, p = 0.80.** The timing adds nothing; returns come from drift and
  diversification.
- **#2 label permutation:** paused at 90/200 runs. Interim: real at the 41st percentile, p ≈ 0.59.
- **#3 synthetic markets:** paused at 2/200 runs.
- Tests #2 and #3 retrain a replica of the v3.1 core loop (`common.py`). Price data is
  gitignored (`research/etf_long_cash/data/`, through 2023-12-29 only), and the Python 3.12
  environment is `research/etf_long_cash/.venv`.
- The published v3.1 report never beat SPY: Sharpe below SPY in 15/15 ETFs, and below
  holding the same ETF in 11/15. No interval showed the model winning.

## Mirrored short model (team idea): why it isn't enough alone
Training "did it go down?" / "short return" on the **same factors** reproduces v4.0, because
those targets are the long targets flipped. A short model only becomes different with new,
direction-bearing factors. Use a cost-beating drop threshold, class weighting and precision
scoring, and run it as a rare overlay on v3.1. Details are in `docs/short_side_diagnosis.md`.

## Git / accounts
Global `git config user.name` changed to "m4nasp06" on Nov 17 (older commits still show
"Technology-TheStringLine"). GitHub credits
commits by email, which maps them to the **m4nasp06** account. `git push` authenticates as
m4nasp06. The `gh` CLI's active account is Technology-thestringline (matters for PRs).

## Python
On the laptop, default `python` (3.14) has pandas but no scipy, and the bare 3.12 install has
neither. Use `research/etf_long_cash/.venv` (3.12 with the full requirements) for the pipeline
and the significance tests.

## Next task: test direction signals inside cash weeks
Split the model's cash weeks by candidate features and check whether forward returns separate
(declines in one group, gains in the other) before retraining anything. Candidates:
1. Trend: price vs its 200-day moving average; sign of the 12-month return.
2. Credit stress: the HYG/IEF ratio and its recent change.
3. VIX term structure: ^VIX / ^VIX3M > 1 means backwardation (panic).
4. Downside vs upside semivolatility.
5. Panic / oversold (VIX spike, low RSI) as a possible rebound or long signal.
Use yfinance locally. Compute features using only data available at each `decision_date`
(no look-ahead). Report results per feature and per ETF, note that the 10/20-session labels
overlap, and keep 2024+ untouched. If a feature separates cleanly, next build a dedicated
short model with a downside target (for example, forward return < −3%), or a trend-filtered
short rule, and backtest it against v3.1 long/cash.

## Working preferences
- Explain quant concepts plainly; I'm still learning (applied math sophomore).
- Be honest about weak results; don't oversell.
- I commit my own work. I need visible commits (Joflay grades on them).
