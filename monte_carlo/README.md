# Standalone Monte Carlo strategy test

One Python module, two dependencies, and two output files. No repository model,
private input pack, manifests, compressed archives, or automatic data downloads.

## Connect the data

Edit the **DATA INGEST** section near the top of [suite.py](suite.py).
`DATA_SOURCE` is intentionally `None`, and `ingest_data()` raises an explicit
error until you connect a source. There is no example-data fallback.

Return a `Data` object containing only:

| Field | Shape | Meaning |
|---|---|---|
| `dates` | days | Actual observed sessions, strictly increasing |
| `names` | assets | Unique, stable security identifiers |
| `close` | days × assets | Consistent total-return adjusted closing prices |
| `eligible` | days × assets | Boolean investment eligibility known at each close |
| `decision_rows` | forecasts | Integer row indices of the original forecast closes |
| `scores` | forecasts × assets | Saved model scores, available by each decision close |
| `groups` | forecasts × assets | Integer matching-group IDs known at that decision |

Group IDs encode the risk characteristics you want held comparable—such as
sector, beta, volatility, size and liquidity. The suite needs only the resulting
IDs, not a second collection of fundamental files. Construct groups before
examining results and use no future outcomes or future quote availability.
At least 20% of eligible assets must belong to groups with multiple members.

Supply historical eligibility and identifiers, including removals. Do not
choose a survivor-only universe. Missing prices outside the investment universe
are permitted only when no holding or order needs that quote. A missing held
price, execution price or entry-volatility window stops the entire experiment;
no name, date or failed variation is silently dropped. Corporate actions must
already be reflected consistently in the wealth-price histories. This compact
engine does not reconstruct cash mergers, claims or broker share events.

## Strategy rules

The **STRATEGY** section is deliberately explicit and independent of v39:

- `target_weights()` buys the highest-score 20% and shorts the lowest-score
  20%, with at least one name per side. Both tails receive half the gross
  exposure, which defaults to 100%. Ties use the supplied security ordering.
- Decisions execute at the next observed close. Fees default to 2 bps on
  actual traded notional; target weights are solved on post-fee account value.
- Stops use entry-frozen 63-session daily volatility, scaled by
  `2 × sqrt(21)` and clipped to a 3–20% adverse move. Same-side additions update
  the average entry basis; the volatility threshold stays frozen until a new
  position or side reversal. A trigger exits at the next close.
- Stopped names stay out until the next original forecast. Remaining holdings
  are not rescaled after exits. A fresh forecast resets the blocked set and
  supersedes old-cycle stop signals at that decision close.
- Ineligible holdings are queued for next-close liquidation using actual quotes.
  The last close liquidates remaining positions and charges fees; new final-day
  entries are skipped. Warm-up history is excluded from performance statistics.

**Replace `target_weights()` and/or edit `Strategy` to match your intended
strategy before interpreting the result.** These reference rules are not a
reproduction of the old repository model. Selection, sizing and execution stay
identical between the actual strategy and every variation. Borrow fees,
financing and cash interest are explicitly assumed zero; no inference about
historical borrowing availability is made.

## Run 10,000 variations

```bash
python -m pip install -r monte_carlo/requirements.txt
python monte_carlo/suite.py --output monte_carlo_report
```

The defaults are 10,000 variations, seed 42, 128 accounts per batch, and a
three-forecast assignment block. Optional arguments:

```bash
python monte_carlo/suite.py --variations 10000 --seed 42 \
  --batch-size 256 --block-forecasts 3 --output monte_carlo_report
```

Within each forecast's risk groups, raw scores are reassigned bijectively across
eligible assets. Identity assignments are allowed. Compatible assignments stay
fixed through a block; changed membership or group labels cause only the
necessary rematching. No future query is used to choose today's assignment.
Random streams are keyed by forecast, group and variation index, so changing
batch size changes memory use, not the simulated strategies.

NumPy processes accounts in batches. Market prices and trailing volatility are
shared; the suite does not copy stock histories or create one process per draw.
Account calculations and metrics use float64. Equity paths retained for plotting
use float32: approximately `4 × evaluation_days × variations` bytes. No
day × asset × all-variations tensor or per-variation files are written.

A local performance check with synthetic test fixtures (315 observed sessions,
128 assets, 10,000 variations) took 8.84 seconds for the backtests and retained
9.57 MiB of equity paths. This is an engineering timing, not a strategy result;
runtime depends on hardware, asset count and test length.

## Outputs

The output directory contains exactly:

- **`report.html`** — actual performance versus the randomized 5th/median/95th
  percentiles; historical equity bands; a CAGR histogram; strategy settings and
  interpretation. Figures are embedded, so the report opens offline.
- **`variations.csv`** — one row for every variation: total return, CAGR, annual
  log growth, Sharpe, volatility, drawdown, turnover, costs and stop count.

The primary comparison counts how many randomized strategies equal or exceed
actual annual net log growth, using `(1 + exceedances) / (1 + variations)`.
This is a conditional placebo frequency. Similar risk groups do not by
themselves establish exchangeability, so it is not automatically a calibrated
p-value for a universal no-predictive-edge hypothesis. Percentile bands describe
historical randomized strategies, not future returns or new independent data.
The suite tests saved signals and does not refit models or correct prior model
selection. Strategy and grouping choices should be fixed before evaluation.

## Test

```bash
python -m unittest discover -s monte_carlo -p 'test_*.py' -v
```

Tests use small synthetic fixtures solely to verify mechanics, including causal
volatility, persistent matching, batch invariance, next-close stops, reentry,
fees, strict missing-data errors and the two-file report. Synthetic fixtures
are never an ingestion or command-line mode.
