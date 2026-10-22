"""Build executed read-only v3 comparison deliverables from saved artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import re
import sys
import textwrap

from build_deliverables import CSS, SCRATCH, emit, html_from_notebook
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_ARMS = ["baseline", "payoff_only", "context_only", "payoff_context"]
LABELS = {
    "baseline": "Frozen v2 control", "payoff_only": "Payoff signal only",
    "context_only": "ETF context only", "payoff_context": "Payoff + context",
    "payoff_buffer": "Payoff + trade buffer", "weekly_target": "Weekly return target",
    "risk_scaled_target": "Risk-scaled return target", "recency_weighted": "Recency weighting",
    "quarterly_refit": "Quarterly refit", "ewma_vol": "EWMA volatility sizing",
    "turnover_buffer": "Signal trade buffer", "trend_10m_fixed": "10-month trend / fixed",
    "trend_10m_vol": "10-month trend / vol", "tsmom12m_fixed": "12-month momentum / fixed",
    "tsmom12m_vol": "12-month momentum / vol",
    "fixed_recipe_cart": "Stable technical CART recipe",
}


class ExperimentReport:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.runs = self.root / "runs/v3_1"
        self.metrics = pd.read_csv(self.root / "reports/v3_metrics.csv")
        self.original = pd.read_csv(self.root / "reports/metrics.csv")
        self.uncertainty = pd.read_csv(self.root / "reports/v3_uncertainty.csv")
        self.audit = json.loads((self.root / "reports/v3_audit.json").read_text())
        if self.audit["status"] != "PASS":
            raise RuntimeError("A passing v3 audit is required before publishing deliverables")
        if self.metrics.duplicated(["ticker", "variant", "cost_bps"]).any():
            raise RuntimeError("Duplicate experiment metrics")
        self.tickers = json.loads((self.root / "runs/v2/run_config.json").read_text())["universe"]
        self.variants = self.metrics.variant.drop_duplicates().tolist()
        self.plan = json.loads((self.root / "config/experiments_v3.json").read_text())
        if len(self.tickers) != 15 or set(self.variants) != set(self.plan["variants"]):
            raise RuntimeError("Expected 15 ETFs and every declared variant")
        self.base = self.at_cost().pivot(index="ticker", columns="variant", values="sharpe").reindex(self.tickers)
        self.buy = self.original.loc[self.original.strategy.eq("buy_hold") & self.original.cost_bps.eq(5)].set_index("ticker").reindex(self.tickers)
        references = pd.read_csv(self.root / "reports/benchmark_diagnostics.csv")
        self.passive_vol = references.loc[references.benchmark.eq("vol_always")].set_index("ticker").reindex(self.tickers)
        self.figures = self.root / "reports/figures_v3"
        self.figures.mkdir(exist_ok=True)

    def at_cost(self, bps=5):
        return self.metrics.loc[self.metrics.cost_bps.eq(bps)]

    def table(self, frame, decimals=3):
        return frame.to_html(index=False, border=0, classes="research-table", float_format=lambda x: f"{x:.{decimals}f}", escape=True)

    def save(self, figure, name):
        figure.tight_layout()
        path = self.figures / f"{name}.png"
        figure.savefig(path, dpi=145, bbox_inches="tight")
        plt.close(figure)
        return path

    def overview_html(self):
        rows = []
        for arm in MAIN_ARMS[1:]:
            delta = self.base[arm] - self.base.baseline
            uu = self.uncertainty.loc[self.uncertainty.variant.eq(arm)]
            rows.append({"Experiment": LABELS[arm], "Higher Sharpe than v2": f"{int((delta > 0).sum())}/15",
                         "Higher Sharpe than ETF holding": f"{int((self.base[arm] > self.buy.sharpe).sum())}/15",
                         "Higher than always-long vol": f"{int((self.base[arm] > self.passive_vol.sharpe).sum())}/15",
                         "Median change versus v2": float(delta.median()),
                         "Holm p < 0.05": int((uu.p_holm < .05).sum())})
        return ("<p><strong>Primary objective: net daily Sharpe above the frozen v2 strategy and the ETF's own holding benchmark.</strong> "
                "All rows use the same ETF-specific dates, next-open accounting and 5 bps one-way costs. "
                "These new experiments reuse already inspected pre-2024 history; all results remain exploratory.</p>" + self.table(pd.DataFrame(rows)) +
                f"<p>Across all {len(self.uncertainty)} new-variant/ETF comparisons, <strong>{int((self.uncertainty.p_holm < .05).sum())}</strong> have approximate Holm-adjusted p below 0.05. This fixed-fit historical bootstrap is not prospective confirmation, and no arm is promoted.</p>")

    def main_table(self):
        frame = self.base[MAIN_ARMS].copy()
        for arm in MAIN_ARMS[1:]:
            frame["Change: " + LABELS[arm]] = frame[arm] - frame.baseline
        frame.insert(0, "ETF holding", self.buy.sharpe)
        return self.table(frame.rename(columns=LABELS).reset_index())

    def passive_control_table(self):
        frame = pd.DataFrame({"ETF": self.tickers, "Always-long 10% vol Sharpe": self.passive_vol.sharpe.to_numpy()})
        for arm in MAIN_ARMS:
            frame["Change vs passive vol: " + LABELS[arm]] = (self.base[arm] - self.passive_vol.sharpe).to_numpy()
        return self.table(frame)

    def main_figure(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
        limit = np.abs(self.base[MAIN_ARMS[1:]].sub(self.base.baseline, axis=0).to_numpy()).max() * 1.1
        for axis, arm in zip(axes, MAIN_ARMS[1:]):
            delta = self.base[arm] - self.base.baseline
            axis.barh(np.arange(15), delta, color=np.where(delta >= 0, "#087e82", "#b75e49"))
            axis.axvline(0, color="#26364d", linewidth=.8)
            axis.set_yticks(np.arange(15), self.tickers)
            axis.invert_yaxis()
            axis.set_title(LABELS[arm])
            axis.set_xlabel("Net Sharpe change versus v2")
            axis.set_xlim(-limit, limit)
            axis.grid(axis="x", alpha=.3)
        return self.save(fig, "primary_axes")

    def all_variants_table(self):
        rows = []
        for arm in self.variants:
            if arm == "baseline":
                continue
            delta = self.base[arm] - self.base.baseline
            uu = self.uncertainty.loc[self.uncertainty.variant.eq(arm)]
            rows.append({"Experiment": LABELS.get(arm, arm), "Positive delta / 15": int((delta > 0).sum()),
                         "Median delta Sharpe": float(delta.median()),
                         "Raw 95% interval above zero": int((uu.delta_sharpe_95pct_low > 0).sum()),
                         "Holm p < .05 / 225 comparisons": int((uu.p_holm < .05).sum())})
        return self.table(pd.DataFrame(rows))

    def ewma_control_table(self):
        ewma = self.at_cost().loc[lambda frame: frame.variant.eq("ewma_vol")].set_index("ticker").reindex(self.tickers)
        control = self.original.loc[self.original.strategy.eq("CART") & self.original.sizing.eq("vol") & self.original.cost_bps.eq(5)].set_index("ticker").reindex(self.tickers)
        frame = pd.DataFrame({"ETF": self.tickers, "CART with 63-session vol Sharpe": control.sharpe.to_numpy(),
                              "Same signal with EWMA Sharpe": ewma.sharpe.to_numpy(),
                              "EWMA minus matched vol control": (ewma.sharpe - control.sharpe).to_numpy()})
        return self.table(frame)

    def all_variants_figure(self):
        arms = [arm for arm in self.variants if arm != "baseline"]
        differences = self.base[arms].sub(self.base.baseline, axis=0)
        limit = max(.15, np.abs(differences.to_numpy()).max())
        fig, ax = plt.subplots(figsize=(16, 8))
        image = ax.imshow(differences.to_numpy(), aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
        ax.set_yticks(np.arange(15), self.tickers)
        ax.set_xticks(np.arange(len(arms)), [LABELS.get(x, x) for x in arms], rotation=55, ha="right")
        ax.set_title("All declared experiments · change in net Sharpe versus frozen v2")
        for i in range(15):
            for j in range(len(arms)):
                value = differences.iloc[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > limit * .65 else "#17283c")
        fig.colorbar(image, ax=ax, label="Change in annualized daily Sharpe", shrink=.8)
        return self.save(fig, "all_variant_sharpe")

    def uncertainty_figure(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True, sharex=True)
        displayed = self.uncertainty.loc[self.uncertainty.variant.isin(MAIN_ARMS[1:])]
        lower = min(0., float(displayed.delta_sharpe_95pct_low.min()))
        upper = max(0., float(displayed.delta_sharpe_95pct_high.max()))
        margin = (upper - lower) * .06
        for ax, arm in zip(axes, MAIN_ARMS[1:]):
            dd = self.uncertainty.loc[self.uncertainty.variant.eq(arm)].set_index("ticker").reindex(self.tickers)
            for i, row in enumerate(dd.itertuples()):
                ax.plot([row.delta_sharpe_95pct_low, row.delta_sharpe_95pct_high], [i, i], color="#7e8b9b", linewidth=1.5)
                ax.scatter(row.delta_sharpe, i, color="#087e82" if row.delta_sharpe >= 0 else "#b75e49", s=25)
            ax.axvline(0, color="#26364d", linestyle="--", linewidth=1)
            ax.set_yticks(np.arange(15), self.tickers)
            ax.invert_yaxis()
            ax.set_title(LABELS[arm])
            ax.set_xlabel("Paired delta Sharpe · raw 95% interval")
            ax.set_xlim(lower - margin, upper + margin)
        return self.save(fig, "uncertainty_primary")

    def cost_table(self):
        rows = []
        for bps in [0, 5, 10]:
            dd = self.at_cost(bps).pivot(index="ticker", columns="variant", values="sharpe")
            for arm in MAIN_ARMS[1:]:
                delta = dd[arm] - dd.baseline
                rows.append({"One-way bps": bps, "Experiment": LABELS[arm], "Higher Sharpe / 15": int((delta > 0).sum()),
                             "Median change versus v2": float(delta.median())})
        return self.table(pd.DataFrame(rows))

    def ledger(self, ticker, variant):
        return pd.read_csv(self.runs / ticker / f"ledger_{variant}_5bps.csv", index_col="Date", parse_dates=["Date"])

    def example_figure(self, ticker):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        colors = ["#26364d", "#087e82", "#a8652f", "#926eac"]
        for arm, color in zip(MAIN_ARMS, colors):
            ledger = self.ledger(ticker, arm)
            axes[0].plot(ledger.index, ledger.nav, color=color, label=LABELS[arm], linewidth=1.4)
            axes[1].plot(ledger.index, ledger.nav / ledger.nav.cummax().clip(lower=1) - 1, color=color, linewidth=1)
        buy = pd.read_csv(self.root / "runs/v2" / ticker / "ledger_buy_hold_5bps.csv", index_col="Date", parse_dates=["Date"])
        axes[0].plot(buy.index, buy.nav, color="#8b99a6", linestyle="--", label="ETF holding", linewidth=1)
        axes[1].plot(buy.index, buy.nav / buy.nav.cummax().clip(lower=1) - 1,
                     color="#8b99a6", linestyle="--", linewidth=1)
        axes[0].set_yscale("log")
        axes[0].set_title(f"{ticker} · isolated feature and payoff experiments, 5 bps one-way")
        axes[0].set_ylabel("NAV · log scale")
        axes[0].legend(ncol=3, fontsize=9)
        axes[1].set_ylabel("Drawdown")
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        for axis in axes:
            axis.grid(alpha=.3)
            axis.set_xlim(buy.index[0], buy.index[-1])
        return self.save(fig, f"example_{ticker}")

    def provenance_html(self):
        manifest = json.loads((self.root / "reports/v3_baseline_hashes.json").read_text())
        return (f"<p><strong>Audit {html.escape(self.audit['status'])}:</strong> "
                f"{self.audit.get('checks_passed', 0):,}/{self.audit.get('checks_total', 0):,} checks passed; "
                f"{len(manifest['files']):,} frozen v2 hashes checked. "
                "The preserved v2 notebook and report remain available as research_notebook_v2.ipynb and reports/research_report_v2.html.</p>"
                "<p>Read saved files: reports/v3_metrics.csv, v3_uncertainty.csv, v3_multiplicity.csv, v3_audit.json; "
                "runs/v3_1 contains corrected forecasts, calibration provenance and independent account ledgers. "
                "Rerunning this notebook only reads saved pre-2024 experiment artifacts.</p>")

    def literature_html(self):
        literature = json.loads((self.root / "reports/literature_review.json").read_text())
        rows = []
        for paper in literature["papers"]:
            title = f"<a href='{html.escape(paper['primary_url'], quote=True)}'>{html.escape(paper['title'])}</a> ({paper['year']})"
            rows.append({"Primary paper": title, "Evidence": html.escape(paper["finding"]),
                         "Practical limit": html.escape(paper["limits_for_this_project"])})
        return ("<p>The papers motivate finite tests and counterchecks; their settings and populations differ from these unlevered ETF accounts. "
                "The frozen v3 plan supplies the actual parameters (including EWMA lambda 0.94 and 26 calibration pseudo-observations); "
                "the standalone literature review's proposed example settings are not claims of exact replication.</p>" +
                pd.DataFrame(rows).to_html(index=False, border=0, classes="research-table", escape=False))

    def context_roadmap_html(self):
        roadmap = json.loads((self.root / "reports/context_source_roadmap.json").read_text())
        rows = []
        for source in roadmap["source_options"]:
            links = " · ".join(f"<a href='{html.escape(url, quote=True)}'>Official source {i+1}</a>" for i, url in enumerate(source["official_urls"]))
            rows.append({"Source family": html.escape(source["id"]), "Qualification state": html.escape(source["status"]),
                         "Availability requirement": html.escape(source.get("availability_contract", source.get("critical_trap", "See roadmap"))),
                         "References": links})
        return ("<p><strong>Current context features are ETF price proxies.</strong> They are not direct nominal/real yields, sector earnings, physical commodity carry or original news revisions. "
                "These future source options are researched and linked, but none supplies additional fitted data in this experiment.</p>" +
                pd.DataFrame(rows).to_html(index=False, border=0, classes="research-table", escape=False))

    def stability_html(self):
        summary = json.loads((self.root / "reports/v3_stability_summary.json").read_text())
        if summary.get("status") != "PASS" or summary.get("actual_run_directory") != "runs/v3_1":
            raise RuntimeError("Corrected-run stability summaries required")
        rows = []
        for arm in summary["primary_four_arm_summaries"]:
            if arm["variant"] == "baseline":
                continue
            for window in arm["windows"]:
                rows.append({"Experiment": LABELS[arm["variant"]], "Window": window["window"],
                             "Higher Sharpe than v2 / 15": window["above_baseline_sharpe_count"],
                             "Higher than passive vol / 15": window["above_causal_vol_reference_sharpe_count"],
                             "Median individual delta Sharpe": window["median_individual_delta_sharpe_vs_baseline"]})
        common = pd.read_csv(self.root / "reports/v3_common_period.csv")
        sub = common.loc[common.variant.isin(MAIN_ARMS[1:]) & common.window.isin(["common_early", "common_late"])]
        table = sub.pivot(index="ticker", columns=["variant", "window"], values="delta_sharpe_vs_baseline").reindex(self.tickers)
        table.columns = [LABELS[arm] + " / " + period.replace("common_", "") for arm, period in table.columns]
        persistent = []
        for arm in MAIN_ARMS[1:]:
            paired = sub.loc[sub.variant.eq(arm)].pivot(index="ticker", columns="window", values="delta_sharpe_vs_baseline")
            eligible = paired.index[(paired > 0).all(axis=1)].tolist()
            persistent.append({"Experiment": LABELS[arm], "Positive in both halves / 15": len(eligible), "ETFs": ", ".join(eligible)})
        return ("<p>The common measurement window is 2012-01-09 through 2023-12-29, with 3,014 sessions for every account. "
                "Early/late windows retain prior positions, opening returns and actual trade costs; normalization changes only the displayed measurement NAV. "
                "These are continuing saved strategies, with no reset, resimulation or new selection at the split. Counts and medians describe independent accounts and are not portfolio performance.</p>" +
                self.table(pd.DataFrame(rows)) + "<p>Persistence across both halves is limited:</p>" + self.table(pd.DataFrame(persistent)) +
                "<p>Per-ETF change versus v2 in each common-window half:</p>" + self.table(table.reset_index()))


def cells():
    import nbformat as nbf
    out = []
    def md(value): out.append(nbf.v4.new_markdown_cell(textwrap.dedent(value).strip()))
    def code(value): out.append(nbf.v4.new_code_cell(textwrap.dedent(value).strip()))
    md("""# Swing Tree · controlled ETF experiments v3.1
    **Payoff signal versus ETF context · 15 independent long/cash studies**

    This executed notebook compares isolated changes with the preserved v2 control. It reads saved results, creates tables and charts, and performs no model fitting, downloads, recipe selection or 2024+ scoring. All experiments use inspected history through 2023; no arm is automatically promoted.
    """)
    code("""from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT/'src/build_v3_deliverables.py').exists():
    raise RuntimeError('Open the notebook from the swing-tree-local project folder.')
sys.path.insert(0,str(ROOT/'src'))
from build_v3_deliverables import ExperimentReport, emit
from IPython.display import HTML, display
V = ExperimentReport(ROOT)
display(HTML(V.overview_html()))""")
    md("""## 1 · What changed and what the comparison means
    The **payoff-only** experiment changes the trading signal: prior chronological out-of-fold forecasts map to shrunk conditional expected net payoff. The **context-only** experiment adds declared ETF-specific, price-based economic context. The **combined** arm applies both changes. The model family, annual v2-selected horizon, target and sizing remain frozen for these three arms.

    Net daily Sharpe uses a zero cash and risk-free rate, 252-session annualization, sample daily volatility and actual-turnover costs. Read drawdown, exposure and compounded return alongside it. All ETF and arm comparisons share each ETF's v2 evaluation dates.
    """)
    code("display(HTML(V.main_table()))\nemit(V.main_figure())\ndisplay(HTML(V.passive_control_table()))")
    md("""## 2 · Payoff calibration and information timing
    The primary payoff rule uses fixed five probability bins for classification and prior out-of-fold prediction quintiles for regression. Bin means shrink toward the prior calibration mean using 26 pseudo-observations. At least 52 matured calibration observations are required; otherwise the original signal is retained and the fallback is recorded. LONG means shrunk expected net payoff above zero; the original sizing rule is retained.

    Calibration forecasts come from earlier annual CART fits with at least 104 matured weekly training rows. Both horizon outcomes and the common 20-session maturity gate must precede the outer cutoff. The historical replay uses the outer-year selected specification as a fixed calibration candidate: its earlier out-of-fold records support calibration, rather than an untouched model-selection performance claim. Uncertainty is outcome dispersion; overlapping labels prevent treating raw counts as independent samples.

    Context uses only declared historically available price relationships with explicit availability flags. It is distinct from revision-sensitive macro, present-day holdings and historical news features, whose availability gates remain separate.
    """)
    md("""## 3 · All declared hypotheses
    The additional experiments separately test weekly targets, risk-scaled targets, recency weighting, quarterly refits, EWMA volatility sizing, trading buffers and simple trend/momentum controls. These are controlled historical comparisons. The table and heatmap describe what happened; selecting a future deployment from them requires a new frozen validation procedure.
    """)
    code("display(HTML(V.all_variants_table()))\nemit(V.all_variants_figure())")
    md("""### Matched volatility-estimator control
    The main all-arm table compares every experiment with the annual v2-selected sizing rule. That mixes a sizing change with the volatility-estimator change whenever v2 selected fixed exposure. The following descriptive comparison instead holds the original CART directional signal and 10% volatility target fixed, comparing EWMA with its 63-session volatility-sized control. No extra hypothesis is promoted from this table.
    """)
    code("display(HTML(V.ewma_control_table()))")
    md("""## 4 · Uncertainty and the full comparison family
    Each ETF jointly resamples the baseline and all fifteen new variants using the same 20-session blocks, with 9,999 replications. The separately frozen inference amendment increases numerical resolution from the initial 500-draw declaration before bootstrap inference. Exact block sums and squared sums accelerate the same paired resampling calculation; tests verify equivalence to full resampled daily returns, including a shortened final block. Error bars are raw paired 95% intervals for Sharpe differences. Approximate centered-bootstrap one-sided p-values receive a Holm adjustment across all 225 new-variant/ETF comparisons.

    These calculations hold fitted strategies fixed. They do not repeat the complete fitting, annual selection, prior research or exploratory choice process. A corrected historical comparison remains insufficient evidence of prospective profitability. The finite bootstrap resolution and dependence assumptions also limit interpretation.
    """)
    code("emit(V.uncertainty_figure())")
    md("""### Common-period and early/late stability
    ETF-specific longest-history results have different start dates. The following additional view aligns all fifteen instruments on one measurement window, then checks early and late halves. Annual saved-ledger results are preserved in [the annual stability file](reports/v3_annual_stability.csv); this descriptive analysis never refits a model or chooses an arm.
    """)
    code("display(HTML(V.stability_html()))")
    md("""## 5 · Cost sensitivity
    Compare each arm with v2 at the same one-way cost assumption. Zero costs are diagnostic; 5 bps are the primary assumption and 10 bps are stress. Each account charges actual traded notional and final liquidation. Cash remains nonnegative, exposures stay between 0% and 100%, and accounts share no capital.
    """)
    code("display(HTML(V.cost_table()))")
    md("""## 6 · All fifteen ETF equity paths and drawdowns
    These charts connect Sharpe changes with full equity paths and drawdowns for every instrument. Every line is a separate normalized account; the charts do not select a winning model.
    """)
    for ticker in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","IEF","TLT","TIP","GLD","DBC","VNQ"]:
        code(f"emit(V.example_figure('{ticker}'))")
    md("""## 7 · Audit, reproducibility and next validation
    The v2 source, forecasts, models, ledgers, metrics and preserved deliverables remain frozen. Audit checks cover matched dates, mature training/calibration outcomes, reserved-period exclusion and cash/cost conservation. Vendor-adjusted synthetic units still approximate total return; the XLF 2016 in-kind distribution requires independent action reconciliation before official validation.

    All results are retrospective and use currently surviving ETFs. Cash yield is zero. Constant costs, simplified fills and public-data revisions do not represent every execution condition. The 2024+ reserve remains unscored, and previously inspected history cannot be relabeled a pristine holdout. Freeze a small candidate procedure before later validation.

    The corrected v3.1 run resolves an engineering branch-order error that initially repeated the 63-session sizing rule in the EWMA arm. All sixteen variants were rerun under the same frozen hypotheses and parameters in a new directory; every non-EWMA metric exactly reproduces the preserved engineering run. See [the engineering amendment](reports/v3_engineering_amendment.json). The displayed results, inference and audit all refer to runs/v3_1.
    """)
    code("display(HTML(V.provenance_html()))")
    md("""## 8 · Primary literature and the next data gates
    Read evidence together with its limitations. Strong historical findings in futures, factor portfolios or a stock panel do not establish profitable timing in each ETF. The current experiments use the fixed local declaration, and the roadmap below records what richer context still requires.
    """)
    code("display(HTML(V.literature_html()))\ndisplay(HTML(V.context_roadmap_html()))")
    md("""The exact mapping from all sixteen frozen experiment IDs to evidence, adaptations and gaps is recorded in [the hypothesis evidence map](reports/v3_hypothesis_evidence.json). Every strategy is an adaptation rather than a claim of direct paper replication. The [inference amendment](reports/v3_inference_plan.json) preserves the numerical-resolution rationale.
    """)
    return out


def verify_and_publish():
    """Publish the already executed comparison only after artifact checks pass."""
    import nbformat as nbf
    from audit_v3 import baseline_changes
    notebook_path = ROOT / "research_notebook_v3_1.ipynb"
    report_path = ROOT / "reports/research_report_v3_1.html"
    notebook = nbf.read(notebook_path, as_version=4)
    original = nbf.read(ROOT / "research_notebook_v2.ipynb", as_version=4)
    nbf.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code):
        raise RuntimeError("Every code cell must execute before publication")
    if any(output.output_type == "error" for cell in code for output in cell.outputs):
        raise RuntimeError("Notebook execution errors prevent publication")
    preserved = notebook.cells[2:2 + len(original.cells)]
    if not all(a.source == b.source and a.cell_type == b.cell_type for a, b in zip(original.cells, preserved)):
        raise RuntimeError("Original notebook cells were not preserved")
    if baseline_changes():
        raise RuntimeError("Frozen v2 content changed")
    metadata = notebook.metadata["swing_tree"]
    if metadata["run_directory"] != "runs/v3_1" or metadata["run_version"] != "v3.1":
        raise RuntimeError("Corrected run metadata required")
    for key, relative in [("frozen_v2_metrics_sha256", "reports/metrics.csv"),
                          ("v3_metrics_sha256", "reports/v3_metrics.csv"),
                          ("inference_plan_sha256", "reports/v3_inference_plan.json")]:
        if metadata[key] != hashlib.sha256((ROOT / relative).read_bytes()).hexdigest():
            raise RuntimeError("Notebook data provenance became stale")
    figures = [ROOT / "reports/figures_v3" / f"example_{ticker}.png" for ticker in ExperimentReport(ROOT).tickers]
    if not all(path.is_file() and path.stat().st_size > 10000 for path in figures):
        raise RuntimeError("All fifteen new instrument figures required")
    page = report_path.read_text(encoding="utf-8")
    if re.search(r"<(?:img|script)[^>]+src=[\"']https?://", page, re.I):
        raise RuntimeError("Report contains external media dependencies")
    for text in ["controlled experiments v3.1", "9,999", "225", "runs/v3_1", "Common-period and early/late stability"]:
        if text not in page:
            raise RuntimeError(f"Missing report content: {text}")
    older_verification = ROOT / "reports/deliverable_verification.json"
    archive = ROOT / "reports/deliverable_verification_v2.json"
    if older_verification.exists() and not archive.exists():
        shutil.copyfile(older_verification, archive)
    shutil.copyfile(notebook_path, ROOT / "research_notebook.ipynb")
    shutil.copyfile(report_path, ROOT / "reports/research_report.html")
    outputs = ["research_notebook.ipynb", "research_notebook_v3_1.ipynb",
               "research_notebook_v2.ipynb", "reports/research_report.html",
               "reports/research_report_v3_1.html", "reports/research_report_v2.html"]
    verification = {"status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    "run_version": "v3.1", "actual_run_directory": "runs/v3_1",
                    "cells": len(notebook.cells), "executed_code_cells": len(code),
                    "execution_errors": 0, "original_v2_code_cells_preserved": sum(cell.cell_type == "code" for cell in original.cells),
                    "all_original_v2_cell_sources_preserved": True, "new_instrument_figures": len(figures),
                    "v2_frozen_hashes_unchanged": True,
                    "visual_QA": "Reviewed primary ablation chart, full variant heatmap and representative equity/drawdown chart. Ablation panels share one axis range; all fifteen new instrument figures are present.",
                    "self_contained_HTML": True, "model_fitting_by_notebook": False,
                    "outputs": {relative: {"sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                                           "bytes": (ROOT / relative).stat().st_size} for relative in outputs}}
    for path in [ROOT / "reports/deliverable_verification.json", ROOT / "reports/deliverable_verification_v3.json"]:
        path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(f"Published verified v3.1 notebook/report: {len(code)} executed cells, 31 original v2 code cells, 15 new ETF figures", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="After successful execution and QA, update the main notebook/report copies")
    parser.add_argument("--publish-existing", action="store_true", help="Validate and publish already executed v3.1 files without rerunning cells")
    args = parser.parse_args()
    if args.publish_existing:
        verify_and_publish()
        return
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    ExperimentReport(ROOT)
    original = nbf.read(ROOT / "research_notebook_v2.ipynb", as_version=4)
    intro = nbf.v4.new_markdown_cell("""# Swing Tree · full research notebook with controlled v3.1 experiments

The complete original v2 study remains below, including all fifteen ETF paths, factors and news demonstrations. Its descriptions of unimplemented stages refer to that preserved initial run. The new controlled payoff/context experiments, additional declared controls, literature and updated audit follow in **V3.1 onward**. Both versions use only pre-2024 strategy evaluation.
""")
    new_cells = cells()
    for cell in new_cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## "):
            cell.source = cell.source.replace("## ", "## V3.", 1)
    top = nbf.v4.new_code_cell("""from pathlib import Path
import sys
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'src'))
from build_v3_deliverables import ExperimentReport
from IPython.display import HTML, display
V3_OVERVIEW = ExperimentReport(ROOT)
display(HTML(V3_OVERVIEW.overview_html()))
""")
    notebook = nbf.v4.new_notebook(cells=[intro, top] + original.cells + new_cells)
    notebook.metadata.update({"kernelspec": {"display_name": "Python 3 (Swing Tree research)", "language": "python", "name": "python3"},
                              "language_info": {"name": "python", "version": sys.version.split()[0]},
                              "swing_tree": {"purpose": "read-only saved controlled experiment comparison", "run_version": "v3.1", "run_directory": "runs/v3_1",
                                             "frozen_v2_metrics_sha256": hashlib.sha256((ROOT/'reports/metrics.csv').read_bytes()).hexdigest(),
                                             "v3_metrics_sha256": hashlib.sha256((ROOT/'reports/v3_metrics.csv').read_bytes()).hexdigest(),
                                             "inference_plan_sha256": hashlib.sha256((ROOT/'reports/v3_inference_plan.json').read_bytes()).hexdigest(),
                                             "reserved_start": "2024-01-01"}})
    kernel_base = SCRATCH / "kernels_v3"
    kernel_dir = kernel_base / "python3"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                                      "display_name": "Python 3 (Swing Tree isolated)", "language": "python"}))
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_base)]))
    def progress(**kwargs):
        print(f"Executed comparison notebook cell {kwargs.get('cell_index', 0) + 1}", flush=True)
    client = NotebookClient(notebook, km=manager, timeout=180, startup_timeout=30,
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
                            on_cell_executed=progress)
    try:
        client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    if any(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code"):
        raise RuntimeError("Notebook contains unexecuted code")
    notebook.metadata["swing_tree"]["executed_utc"] = datetime.now(timezone.utc).isoformat()
    notebook_path = ROOT / "research_notebook_v3_1.ipynb"
    nbf.validate(notebook)
    nbf.write(notebook, notebook_path)
    report_path = ROOT / "reports/research_report_v3_1.html"
    html_from_notebook(notebook, report_path)
    page = report_path.read_text(encoding="utf-8").replace("local run v2", "controlled experiments v3.1").replace("Swing Tree · measured ETF outcomes", "Swing Tree · controlled ETF experiments v3.1")
    report_path.write_text(page, encoding="utf-8")
    if args.publish:
        verify_and_publish()
    print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells; v3 notebook/report saved; published={args.publish}", flush=True)


if __name__ == "__main__":
    main()
