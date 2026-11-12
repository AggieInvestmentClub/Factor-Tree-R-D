PORTFOLIO STATISTICAL VALIDATION — v40 — through November 11, 2026

Open index.html in a browser. Figures and tables use relative links; keep the
folder structure intact. No server, account connection or network is needed
to view the report. PNG and SVG versions of every figure are included.

WHAT IS INCLUDED
12 figures; full-precision CSVs; simulated terminal return samples; saved
forecast evaluation rows; analysis scripts; protocol, source hashes and QA.
The two model versions are unchanged. No refit, selection or promotion.

REPRODUCE THE HTML FROM PACKAGED RESULTS
python src/report.py

REPRODUCE THE FULL ANALYSIS
python src/run.py
python src/verify.py

Full analysis additionally requires the original sibling research folders
v31_growth_long_bias, v38_position_exits and v39_forward_validation plus
their original dependencies (v17/v23/v28/v29/v34/v36 as imported by the saved
portfolio code). The original data are not repackaged. Exact source hashes
are in protocol.json. The requirements.txt records the analysis environment.
Some source code retains machine-specific research paths. Run in the original
workspace, or explicitly remap those paths while preserving source hashes.

INTERPRETATION
The equity curve links v38 (2007–2023) and v39 (2024–2026), including cash
restart at the 2024 boundary. This is not an uninterrupted account replay.
The forward era was previously examined by an older model; not pristine OOS.
Earlier held-quote outages, aborted batches and conditional corporate-action
valuations remain in the source accounts. Full-history provisional valuation
sessions: 51 benchmark / 50 stop. Forward period: 15 per account, no held-price
gaps. See tables/source_audit.json for original source issues.
Base accounts include 2 bp per side; borrow, funding and cash yield are zero.
Statistical inference is conditional on these account and source assumptions.
Future scenarios assume the empirical return distribution persists and do
not integrate uncertainty about the underlying expected return.

No raw vendor price history or credentials are included. Aggregate report and
derived evaluation files are packaged for review; no upload was performed.
