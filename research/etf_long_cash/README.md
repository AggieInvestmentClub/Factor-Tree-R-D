# ETF long/cash research infrastructure

The current saved model is **v3.1**, with separate long/cash accounts for 15 ETFs.
The report concentrates on the combined **payoff-aware signal + ETF-context**
procedure. Net daily Sharpe after costs is the primary performance measure.

## Start here

- [Performance report and results](focused_payoff_context/README.md)
- [Executed statistical notebook](focused_payoff_context/payoff_context_review.ipynb)
- [Saved report inputs and detailed statistics](focused_payoff_context/data/)
- [Research sources](src/), [frozen configurations](config/), [synthetic tests](tests/)
- [Full declared hypothesis results, stability and evidence review](reference/study_v3_1/)
- [Base factor definitions](reference/feature_registry.csv) and [ETF-context definitions](reference/etf_context_registry.csv)

Project .gitattributes preserves exact artifact bytes so recorded hashes survive checkout on different platforms.

The notebook contains 139 executed code cells and 61 embedded figures. It reads
the curated saved results without fetching prices or fitting a model. Its CSVs,
helpers and figures are committed together so the report remains portable.
There is no portfolio allocation, shorting, leverage, options or live execution.

## Run the report or checks

Use an isolated Python 3.12 environment. From this directory:

    python -m pip install -r requirements-core.txt
    python -m unittest discover -s tests -v
    python scripts/validate_published_report.py

Open focused_payoff_context/payoff_context_review.ipynb in Jupyter/VS Code and
select that environment. Run All from the notebook's directory. The saved
HTML report can also be downloaded and opened locally.

The focused src/diagnostic_audit.py independently checks saved accounting,
calendar returns, drawdowns, forecasting/calibration, weekly intervals and
benchmark comparisons. The compact report requirements also live beside the
notebook. requirements-lock.txt records the full original research environment,
including optional news/FinBERT dependencies; it is not needed just to inspect
the saved notebook.

## Training/data pipeline and dependencies

src/data_ingest.py archives/normalizes ETF and public macro sources.
src/research.py implements the expanding v2 CART control and bounded RF/XGB
challengers. src/etf_context.py builds the causal ETF-specific price-context
pack. src/experiments_v3.py performs the frozen v3.1 experiments and calibrates
expected payoff from earlier matured out-of-fold forecasts. Source/config
snapshots match the local study; no saved predictions were retrained for upload.

Raw downloads, normalized market histories, article text, model checkpoints,
pickles and complete fitted-run directories are local and ignored.
Fresh public-source downloads can differ from the historical vendor snapshot.
The committed curated inputs reproduce the statistical report, not the complete
training fits by themselves.

For a separate, explicitly initiated research rerun, use this dependency order:

1. Prepare local data/normalized/prices/ and the optional macro archive with
   src/data_ingest.py. A historical-only example is
   python src/data_ingest.py --cutoff 2023-12-31; install acquisition extras
   from the full lock as needed. This invokes external data requests.
2. Create local reports/, then run python scripts/build_feature_contract.py
   after ingestion has generated its macro diagnostic file.
3. Run python src/research.py to create the local runs/v2/ baseline and
   reports/metrics.csv; existing completed forecasts are protected from overwrite.
4. Run python src/etf_context.py against the local normalized source histories.
5. Run python src/experiments_v3.py; it requires the v2 forecasts, annual
   selections, all cost ledgers and metrics. Its output is runs/v3_1/.
6. The saved-result audit/report builders additionally require their baseline
   hash snapshots, original report artifacts and metadata. Their reference
   snapshots are not a substitute for recreating those full local dependencies.

Declare a new version/output path before changing a completed study. Do not
silently reuse the frozen v3.1 name for later model changes. FinBERT and sector
news acquisition code is provided, but neither supplied eligible fitted
historical factors in the reported model.

## Research interpretation

All strategy evaluation ends in December 2023. **2024 onward remains reserved**.
Weekly completed-close decisions execute at the next actual open. Accounts use
fractional adjusted units, no second dividend credit, zero cash yield, no leverage,
5 bps one-way actual-turnover fees and terminal liquidation. Zero/10 bps stresses
are saved. A forecast's 5/10/20-session horizon is not its realized holding period.

The combined procedure improves baseline Sharpe in 10/15 instruments on matched
instrument-specific histories, with median difference about +0.040. These are
retrospective exploratory results; none of the parent study's 225 new comparisons
passed its Holm 5% adjustment. New ETF/SPY benchmark intervals in the focused
report are post-hoc fixed-fit descriptions, not additional confirmation tests.
Macro/news timing, vendor revisions and the XLF 2016 in-kind distribution remain
qualifications. Read the report's definitions and limitations with its statistics.

The broad historical report builder retains its original scratch-path assumption (a work/report-runtime folder above the research project). Use the portable focused notebook for the published report; original export helpers in its reference_sources folder are provenance snapshots.

The older stock/put wording in existing team documents is historical context;
this addition implements the later ETF long/cash research scope.
