# Data contract and qualification

## What the data represents

The stock universe changes through dated S&P 500 membership and sector
reconstructions. It is approximately 500 concurrent names, not a fixed list
chosen in 2007 and held forever. Historical delisted identifiers are retained.
Membership and sector anchors are reconciled against joiner/leaver events.
These are reconstructed histories from later captures, not a vintage database
proving exactly what a vendor displayed at each historical decision time.

Prices include qualified adjusted histories, raw cash-conversion quotes,
segment flags, event terms and delayed distribution claims. Source approval
preserves 1,085 prior price prefixes exactly. Its forward review records zero
interior quote gaps, zero disagreements with the independent direct-quote
checks, 18 terminal events, 18 noncash events and 38 verified splits.
These are the frozen source qualification results, not a new independent
vendor recapture performed for this handoff.

## Private input pack

`data_contract/input_manifest.json` lists every required file and exact hash.
`replay` supplies the qualified price/event package, training arrays, daily
eligibility/betas, checkpoints, sealed metadata and initial target.
`rebuild` additionally supplies the original pre-2024 feature/label/gate/risk
artifacts and dated memberships/sectors. Both profiles use the same data.
The manifest has no machine-specific source locations.

The local exporter used to assemble the data pack retains a private
`source_locations.json` outside Git. This file is not needed to run the package
and should not be shared or committed. Supply only the `current/` and `prior/`
input trees to an appropriately authorized collaborator. Raw vendor captures,
API credentials and workstation configuration are not part of the Git package.

| Data | Alignment / required interpretation |
|---|---|
| `data/assembled/prices_candidate.pkl` | Dictionary of issue-level dated frames; qualified quotes require `q == 1`; preserve raw and adjusted prices and segment IDs |
| `events.csv`, `claims.json`, `noncash_census.csv` | Effective event dates, converted payoff terms, claim settlement clocks and exposure audit census |
| `calendar.csv` | Observed session clock; no fabricated weekends or holiday returns |
| Membership and sector snapshots | Dated eligible identifiers/sector assignments; identifiers are not automatically stitched by reused ticker |
| Model `metadata.json` | Ordered names, decision dates, label-end dates, feature columns and full daily calendar |
| `x.npy` | Query x ordered-name x 58 causal features |
| `mask.npy`, `beta.npy`, `gate.npy` | Prediction eligibility, contemporaneous beta and mature-history gate |
| `returns.npy`, `complete.npy` | H21 training outcomes and whole-query maturity, not prediction-universe filters |
| `centered.npy`, `diag.npy` | Query x 63-day x name return history, plus diagonal variance |
| `orth.npy`, `corr_orth.npy`, `factor_loadings.npy` | Neutrality bases and top-three correlation-factor loadings |
| `daily_beta.pkl`, `daily_eligible.pkl` | Daily risk repairs and eligibility on the account calendar |
| `qualified_returns.npy` | Daily qualified returns used by entry-volatility stops |

Do not zero-fill missing observed returns. Internal tensor padding of inactive
names is allowed only behind the eligibility mask; the trainer asserts that
all active training outcomes are finite. The account adapter only sets
missing distribution-trade turnover to zero on rows without qualified quotes;
it does not fill a price or return.

## Remaining qualifications

Both forward accounts have **15 provisional valuation sessions**:

- 13 Lennar/Millrose claim sessions from January 21 through February 6, 2025.
  Claims use an explicit zero-mark scenario before settlement at the first
  regular-session raw open on February 7. These are not observed interim NAVs.
- Walgreens contingent rights on August 28, 2025 and Hologic contingent rights
  on April 7, 2026 remain conditional cash-only valuations.

The arithmetic and source qualification checks pass, but the account audit's
`validated_performance` remains **false**. Packaging must not erase that flag.
Unsupported later actions for CMA and QRVO are excluded by eligibility and
protected by an account exposure assertion. No forward held-price outages,
margin calls or aborted trade batches were reported in the frozen accounts.

Full-history linked results inherit additional earlier issues: 51 provisional
sessions for the benchmark and 50 for the stop, plus earlier held-price
outages/aborted batches. Read `verification/full_history_source_audit.json`
before applying full-history inference. The v39 rerun validates only the
specified forward account against its frozen sources.

## Fresh handoff checks

`verification/reproduction.json` records the portable rebuild comparison.
`verification/data_check.json` records file integrity, array/date alignment,
model maturity, and reference-series checks. `verification/package_check.json`
records the publication inventory, syntax/hash checks and credential-pattern
scan. These establish reproducibility and packaging hygiene, not freedom from
survivorship, vintage, model-selection or execution assumptions.
