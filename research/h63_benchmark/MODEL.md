# Frozen H63 specification

## Universe and timing

The storage universe has 1,008 historical stock identifiers, with roughly 500 eligible at a time. Membership changes through history; it is not a fixed list of today's surviving stocks. Daily membership, quote availability, feature history and risk-history requirements determine eligibility using information available at the decision close. The full account covers January 4, 2007–November 11, 2026 (4,996 sessions). Annual models use a rolling preceding ten years, with fully mature labels before January 1. Early fits are sparse: the 2007 H63 model has only five qualified training queries.

Decisions refresh on the original 21-session calendar grid. **H63 means a 63-session prediction target, not a 63-session holding commitment.** Decisions use the close at t and execute at the next session's close. The forward account starts January 2, 2024 from cash. Its initial target uses the December 8, 2023 feature row with the 2024 models, then December 29 membership/beta repair; the covariance remains from the original feature row. This boundary convention is preserved exactly.

## Stock state: 58 features

The original 36-column state contains 22 cross-sectional technical measurements, three availability indicators, and eleven sector ETF indicators. The technical measurements cover returns over 1/5/21/63/126/252 sessions; 21-session volatility; 63-session market beta; distance from a 21-session moving average; 63-session drawdown; RSI; Bollinger z-score; typical-price VWAP distance; ATR as a percentage of price; volume ratio and price/volume correlation; and separate overnight/intraday cumulative returns and spreads over 5/21 sessions.

The 36 columns receive fast EWMA smoothing (alpha .25). The first 22 also receive slow smoothing (half-life 21 sessions), giving 58 inputs. States update on eligible daily observations and reset when a name becomes ineligible. `data_contract/features.json` gives the exact ordered schema. The raw feature recipe is preserved under `provenance/recipes/stock_features_v39.py` and the state-building recipe under `build_arrays.py`. Availability handling is not permission to fill unavailable prices or training outcomes.

## Ranking: learned nonlinear trees

Eight decision-close context measurements are appended to the 58 stock inputs: 20-session downside EWMA volatility; its difference from the 60-session measure; the 60-minus-120 difference; 120-session EWMA mean log market return; cross-sectional dispersion of 21-session returns; median 21-session market-residual volatility; positive-return breadth; and mean pairwise 63-session stock correlation. Downside measures use half-lives 20/60/120, minimum 126 observations and negative market log returns. Context is common to stocks on each date; trees can interact it with stock-specific inputs. H63 does not include a fitted jump-state classifier.

The target is the stock's qualified next-close-entry 63-session total return minus its decision-date market beta times the corresponding SPY return. Each date's eligible outcomes are converted to five relevance buckets by average ranks. LightGBM LambdaRank learns the within-date ordering, with linear label gains 0–4. Scores are ordinal, not calibrated expected percentage returns.

Fixed recipe: 100 trees, learning rate .05, 7 leaves, maximum depth 3, minimum 200 observations per leaf, L2 10, top-100 LambdaRank truncation, query-normalized lambdas, deterministic single-thread seed 20261119. Annual training uses the common complete-query census from the original H5/H21/H63 experiment. Every required outcome must be mature and qualified; prediction eligibility never uses future label availability. Overlapping 63-session labels do not create independent observations.

## Capital policy: retained separate learner

An earlier portfolio policy still supplies the **distribution of requested stock weights and cash demand**. It uses the 58 stock states plus 32 fixed cosine transforms, with 90 learned linear readout coefficients. Frequencies/phases are frozen random features (seed 20261111); the readout is trained with Adam for 100 epochs, learning rate .01 and gradient clipping 1, in float64.

Its objective maximizes annualized net log growth in a 21-session portfolio replay, including holding drift, the .20 L1 trading buffer and 2 bp one-way fees, plus L2 regularization. Prior two-year daily-account validation selects .1 or .01 regularization against a broad-long control; sparse or disqualified validation falls back to .1. The annual H63 tree ranker and this 21-session capital policy therefore have different objectives and horizons. Both checkpoints are necessary to reproduce H63.

## Rank-to-position mapping

For each eligible cross-section, sort the capital-policy scores, sort stocks by the H63 ranker, and assign the largest old score to the highest-ranked stock, continuing downward. Stable sorting fixes ties. This preserves the original raw score distribution, positive-score count and requested cash demand while changing stock identities.

Raw requested weight is `max(3 × reassigned_score / eligible_count, 0)`. Negative requests become cash. There is no normalization forcing full investment. Apply one common contraction, never an upward expansion, so targets satisfy all of:

- Forecast annual volatility at most 12%.
- Each name at most 15% of NAV.
- Gross at most `min(eligible_count / 500, 1)`.
- Net invested exposure at most 99%.
- Absolute estimated portfolio market beta at most 1.
- Long positions and cash only; no borrowing or short stock positions.

Forecast covariance is 50% full sample covariance plus 50% diagonal covariance from the trailing 63 sessions, annualized by 252. Composition changes can alter the final risk contraction even when the raw weight distribution is preserved. The 2009 low-investment behavior remains part of the benchmark.

## Sector, regime and long-bias controls

Sector indicators affect ranking, and joint return covariance affects total risk. H63 has **no separate hard sector allocation target**, no sector-neutrality rule, no PCA neutrality requirement, no fixed beta overlay and no volatility stop. Its continuous context can condition ranking on market conditions. The long bias follows from nonnegative weights and retained cash rules. Realized beta around .53 in 2024 onward is an outcome, not a fixed .53 target or guaranteed ceiling. Experimental constraints anchoring challengers to H63 sectors/beta were comparison controls, not H63 rules.

## Execution and account

The engine compares desired weights with drifted holdings. Ordinary rebalance requests must cross a portfolio-wide .20 L1 distance; mandatory guards can override the buffer. Daily checks include gross above 1.1, net/beta exceeding mandate bounds by .05, unknown held beta, ineligible held stocks and forbidden short exposure. Repairs execute next close, subject to actual quote availability. Scheduled 12% volatility and 15% name limits do not guarantee those bounds between trades or intraday. H63 does not use the later daily volatility-sizing overlay.

The account tracks cash, shares, transaction costs, eligible quotes, corporate settlements and distribution claims. Missing execution prices can abort an order batch. It uses theoretical fractional holdings, modeled close fills and zero cash interest. Historical holdings liquidate at year-end 2023 and the forward account restarts; live trading would not repeat this artificial research boundary.

## Interpretation

H63 has lower historical market exposure and drawdown than SPY, but lower CAGR in both reported windows. The stock-specific ranking signal remains modest, and previous research did not establish robust material multifactor alpha. Its successful reproduction is evidence of implementation consistency, not proof of future performance. See numerical diagnostics and qualifications in `reference/` and `DATA.md`.
