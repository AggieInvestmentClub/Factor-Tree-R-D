# Data boundary and qualification

## Public versus private

This release follows the repository's v39 convention: vendor-supplied security histories, prepared per-stock features/labels, daily eligibility and risk panels, and corporate-event input records remain in an authorized **private** input pack. Git contains code, model checkpoints, strategy-level return data, account diagnostics, charts, schemas, hashes and verification evidence. No vendor credentials, desktop configuration, raw cache, research environment or unrelated experiment is included.

The input manifest enumerates every private file, byte count and SHA-256. Do not substitute a current constituent list, forward-fill missing quotes, fabricate delisting returns or replace qualified labels with zeros. Source-qualified corporate actions and identifier history are economically meaningful. A clean file manifest does not certify vendor history as original-vintage point-in-time data.

## Timestamps and validation

- Feature/risk inputs end at the decision close. Orders execute strictly later, at next close.
- Annual ranker training requires the whole common query to be complete and every H63 end date before January 1. Capital-policy H21 labels likewise mature before the fit cutoff.
- Training label availability does not determine prediction-time eligibility.
- Private prepared arrays retain the original floating-point representations. Published checkpoints and reconstructed scores/targets are compared to their original saved values.
- Full and forward daily ledgers are compared field by field, including costs, exposure and quality flags.
- Empty distribution-turnover metadata is normalized to zero only on rows without qualified quotes. No missing price or return is imputed.

## Preserved issues

Full-sample performance includes 33 sessions whose market P&L is not fully validated and two aborted order batches; 15 provisional sessions are in 2024 onward. Interim delayed-distribution entitlements can carry an explicit zero-mark scenario until observed settlement. Such a value is conditional, not an observed tradable NAV. The forward Lennar distribution claim is one example. Original issues, dates and explanations remain in `reference/*_complete.json`, with daily flags in the ledgers and held-price issues in `*_marks.json`.

Accounting identities pass despite these source qualifications. No provisional or adverse session has been removed to improve metrics. The same universe, cash rules, execution assumptions and risk limits are retained. In particular, publication does not silently repair the 2009 cash behavior.

## Reproduction scope

Supported: private prepared arrays → annual models/scores → H63 desired targets → original daily account; plus public daily-return and chart analysis. Capital refits use recorded validation-selected penalties. Upstream raw acquisition and reconstruction recipes are retained for inspection, but are not advertised as independently runnable vendor download tools. See `README.md` for supported commands and `provenance/release_verification.json` for checks actually run.

Data end: November 11, 2026. Nothing in this release claims coverage through the upload date or a newly untouched 2024 holdout. Live use requires a separately tested data updater, state persistence, corporate-action process, broker adapter and execution reconciliation.
