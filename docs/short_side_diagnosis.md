# Why the short side fails: diagnosis

*Manas, Nov 2026. Reproduce with `python scripts/short_side_diagnosis.py` from the repo root.*

## The question

v4.0 turned the v3.1 trees' "cash" decisions into possible shorts, and Sharpe fell in 14 of 15 ETFs.
Shorts can only make money if the ETF tends to **fall** after the model exits. The saved v3.1
forecasts include what actually happened afterward, so we can check this directly. v4.0 reuses
these same annual trees.

## Findings

Data: 11,955 matured weekly decisions, 2008–2023, gross forward returns converted to a
per-week equivalent.

**1. After a "cash" decision, ETFs kept going up, by about as much as after a "long" decision.**

| Model said | Weeks | Avg forward return | Fell afterward |
|---|---|---|---|
| Long | 8,776 | +14.7 bp/wk | 43.0% |
| Cash | 3,179 | +16.6 bp/wk | 43.5% |

A short needs the "fell" share to be well above 50%. It's 43.5%, the same as the base rate.

**2. This holds in nearly every ETF.** After cash decisions the forward return was positive in 14
of 15 ETFs. The worst cases are XLK (+71 bp/wk), XLY (+57), XLU (+39), XLF (+34) and XLV (+33).
Shorting those weeks bets against strong upward drift. **DBC is the only ETF where cash weeks
were followed by declines (−6 bp/wk), and it's also the only ETF with negative drift overall.**

**3. The model's bearishness slightly predicts rebounds.** The rank correlation between the
model's expected payoff and the actual forward return is slightly **negative** in 12 of 15 ETFs
(median −0.03; XLK −0.17). The most bullish quintile had the *lowest* forward return (+3 bp/wk).

**4. Cash weeks are volatile, not bearish.** Measured in z-scores within each ETF, returns
after cash decisions are more dispersed in **both** directions. The 5th percentile is −1.68
versus −1.56, and the 95th is +1.63 versus +1.43. The model exits when things are uncertain.
That cuts drawdowns, which is where the long-only edge comes from, but uncertain periods
include sharp rebounds, and a short absorbs those.

## Conclusion

The long/cash trees were never trained to predict declines. "Cash" means "uncertain," not
"down." Turning it into a short bets against positive drift, and in volatile periods it also
takes on the rebound risk. That matches v4.0, where short episode win rates were mostly
17–40%. A working short side needs its own signal.

## Follow-up: is going to cash worth it? (volatility check)

*Reproduce with `python scripts/cash_week_volatility_check.py` (needs only pandas).*

Same 11,955 decisions. For each group, the forward weekly return's mean, standard deviation and
"return per risk" (mean ÷ std, a per-decision Sharpe, not annualized):

| Model said | Weeks | Avg return | Std | Return per risk | Worst 5% avg |
|---|---|---|---|---|---|
| Long | 8,776 | +14.7 bp/wk | 203 bp | 0.072 | −509 bp |
| Cash | 3,179 | +16.6 bp/wk | 223 bp | 0.074 | −540 bp |

- Cash weeks were more volatile in 11 of 15 ETFs (sign test p≈0.12): a tendency, not a clean rule.
- Cash weeks had worse return per risk in only 8 of 15, a coin flip.
- The model's return per risk (earning only in long weeks) beat always-long in only **3 of 15**
  ETFs (DBC, XLE, IEF; p≈0.035). Median 0.061 vs 0.082.

**Reading:** the filter cuts exposure but does not pick worse-than-average weeks to skip. v3.1's
lower drawdown mostly comes from being in cash ~27% of the time, not from timing. The ETFs where
it hurts most are those whose cash weeks did best (XLK, XLU, XLV, XLF). Same caveats as above:
overlapping labels, gross returns, ETFs not independent, so the p-values overstate the evidence.

**Confirmed by a placebo test** (`wip/manas_significance_tests/`, test #1): v3.1's real
decisions were compared with 10,000 random-timing versions that have the same cash weeks
per ETF, the same run lengths and the same trade count. v3.1's portfolio Sharpe (0.567) sat at
the 20th percentile of random timing (median 0.622), p = 0.80. Retraining tests (label
permutation, synthetic markets) are in progress there.

## Why a mirrored short model would not help

Proposal discussed: train a short model the same way as the long one, with "did it go down over
the next 10–20 sessions?" (classification) and "short return" (regression) as targets.

On the **same factors**, this reproduces v4.0. "Down" is the complement of "up", so the
classifier learns P(down) = 1 − P(up) with the same splits; the short return is the long return
with a minus sign, so the regression tree is the same tree inverted. It would short exactly the
weeks the long model calls cash, which are followed by gains 56.5% of the time.

What makes a short model different is **new information, not a new label**. The same targets on
direction-bearing factors (trend, HYG/IEF, VIX term structure, downside volatility) would be a
real test. Also: label "down" only when the drop beats costs (e.g. < −0.5 to −1%), weight the
rarer class, judge it on precision of short calls, and use it as a rare overlay on v3.1 long/cash.

## What to test next

These are hypotheses, each to be tested on pre-2024 data only, with the 2024+ period still held back.

1. **Train a dedicated short model.** Use a target like "forward return below −X%" (an
   asymmetric, downside-only label), not the long/cash classifier reused.
2. **Short only against the trend.** Allow shorts only when the ETF's own trend is down
   (for example, price below its 200-day moving average, or negative 12-month momentum).
   DBC suggests shorts work only where drift is negative.
3. **Add bearish factors.** Candidates are credit spreads, VIX term structure, market
   breadth and the news-sentiment idea. The current trees are mostly technical and
   relationship features.
4. **Hedge instead of short.** Since cash weeks have fat tails on both sides, defined-risk
   protection (puts, or put spreads, close to Abhinav's put concept) may suit better than
   outright shorts, because they cap the loss when the ETF rebounds.

Caveats: weekly decisions with 10- and 20-session labels overlap, so row counts overstate
independent observations. These are gross returns with no borrow costs. The analysis is
descriptive; no significance is claimed.
