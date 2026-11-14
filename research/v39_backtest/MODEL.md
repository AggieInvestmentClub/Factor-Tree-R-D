# Model and execution specification

The benchmark is `v36 fixed_corr_long50 = v38 none = v39 benchmark`.
The challenger adds the frozen `v38 vol2sigma` position exit. No stop threshold,
bias cap or architecture was selected using the v39 forward comparison.

## Prediction and learning

- Cross-sectional stock selection using 36 price-derived input columns and
  dated sector flags; the exact list is in `specification/features.json`.
- Daily causal fast EWMA (alpha 0.25) of all 36 columns plus a slow EWMA
  (21-session half-life) of the first 22 produces 58 state features.
- A linear score plus a fixed 32-feature random cosine map. The cosine
  frequencies/phases use seed 20261111. Trainable linear readouts use float64.
  There are no earnings surprises, analyst revisions or filing predictors in
  this retained benchmark.
- Forecasts every 21 observed sessions, retaining the original calendar anchor
  of January 3, 2007. Data before that date provide training/warm-up history.
- Each annual model uses a rolling 10-year training window. Labels must end
  strictly before the relevant training boundary. The prior two years select
  L2 penalty 0.1 or 0.01 using the repaired daily account's log growth.
  The selected penalty then refits the full mature training window.
- Adam, learning rate 0.01, 100 epochs, gradient clipping at 1.0.

The optimized loss is negative annualized log growth of a 21-session portfolio
surrogate plus `penalty / 2 * sum(parameter**2)`. The surrogate includes cost
of turnover, carried holdings, a 0.2 L1 no-trade buffer and risk/eligibility
repairs. Annualization is 12 times the mean period log growth. This is an
approximation to daily execution, not differentiation through every account
event. Hyperparameter selection uses the actual daily engine, reducing that
mismatch. See `engine/current/v31_growth_long_bias/src/learner.py`.

## Gate and allocation

The efficacy gate is a separate H5 Huber-ridge model (penalty 100, 10 IRLS
iterations, rolling 10-year refit). Its signal assessment uses 252 strictly
mature whole-date forecast queries and exponentially weighted IC (63-session
half-life). Mature query history is separate from contemporaneous stock
eligibility. The longer 21-session portfolio horizon and H5 gate are deliberate
frozen design choices; they were not unified during packaging.

Risk uses a 63-session history and 50% sample / 50% diagonal shrinkage. The
stock-selection sleeve is projected against dollar, SPY-beta and the top three
correlation factors. The separate market basket supplies controlled long bias
up to 0.50. Both share a 12% annualized forecast-volatility budget.
The learned neutral portfolio is trained before the correlation projection and
long-bias overlay; the training objective is not an exact direct optimization
of the final biased daily account.

| Constraint | Frozen setting |
|---|---|
| Gross | 3 times active-stock count / 500, bounded at 3 |
| Individual weight | 0.15 absolute |
| Turnover buffer | 0.2 L1 |
| Gross drift guard | 3.3 |
| Dollar/beta repair tolerance | 0.05 under the respective engine policy |
| Costs | 2 bps per side |
| Borrow fees / financing / cash yield | Zero |
| Cash borrowing | Disabled |

## Information and execution clock

Features and risk at decision close `t` use observations through `t`.
An order executes at the next observed close `t+1`. An H21 label measures
return from that execution close to `t+22`. A training row is usable only
after the whole required query's labels mature before the refit boundary.
The eligibility mask used to predict does not require a future return.

Both accounts begin in cash January 2, 2024. The final pre-2024 scheduled target
is carried to the December 29 decision close for the first executable order.
The original cadence continues, rather than restarting a fresh 21-day clock.
The final session retains the inherited liquidation/cost convention.

## Volatility-scaled position stop

At an actual entry, freeze daily volatility from 63 qualified sessions.
The adverse-return threshold is `clip(2 * sigma_daily * sqrt(21), 0.03, 0.20)`.
The rule observes adjusted closes, triggers after a close, and executes the
exit at the next close. It is not an intraday stop or a guaranteed fill at the
threshold. An exited name remains blocked until the next original forecast.
The surviving targets undergo the inherited risk-preserving reprojection.
There is no profit target or trailing-stop rule in this challenger.

## Implementation map

| Source | Responsibility |
|---|---|
| `current/v39_forward_validation/src/portfolio.py` | Forward refits, targets and two accounts |
| `current/v39_forward_validation/src/features.py`, `labels.py`, `gate.py`, `build_arrays.py` | Causal data pipeline |
| `current/v29_nonlinear_portfolio/src/policy.py` | Score network, tensors, neutral sizing |
| `current/v31_growth_long_bias/src/learner.py`, `allocation.py` | Log-growth surrogate and long-bias allocation |
| `current/v34_stock_risk/src/risk_learning.py` | Correlation-factor loading/projection |
| `current/v38_position_exits/src/exits.py`, `exit_account.py` | Exit decisions and repaired daily execution |
| `prior/v17_long_history/src/` and `prior/v5_1/src/` | Historical feature, risk and account primitives |

Paths in this table are relative to `engine/`. Annual checkpoint JSON records
include the selected penalty, validation utilities, training query hashes,
maximum label-end dates and epoch traces.
