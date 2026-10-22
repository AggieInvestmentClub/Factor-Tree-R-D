"""Export the saved payoff/context procedure as a portable, executed notebook."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from importlib.metadata import version
from datetime import datetime, timezone

import nbformat as nbf
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE / "outputs/swing-tree-local"
DEST = PROJECT / "focused_payoff_context"
TICKERS = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","IEF","TLT","TIP","GLD","DBC","VNQ"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    for folder in ["data", "src", "figures", "reference_sources", "data/model_rules"]:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    source_files = []
    ledgers, forecasts, usages, calibrators, costs = [], [], [], [], []
    for ticker in TICKERS:
        run = PROJECT / "runs/v3_1" / ticker
        specifications = [
            (run / "ledger_payoff_context_5bps.csv", "payoff_context"),
            (PROJECT / "runs/v2" / ticker / "ledger_buy_hold_5bps.csv", "etf_hold"),
            (PROJECT / "runs/v2" / ticker / "ledger_SPY_context_5bps.csv", "spy_hold"),
        ]
        for path, strategy in specifications:
            d = pd.read_csv(path)
            assert pd.to_datetime(d.Date).max() < pd.Timestamp("2024-01-01")
            d.insert(0, "strategy", strategy)
            d.insert(0, "ticker", ticker)
            ledgers.append(d)
            source_files.append({"path": str(path.relative_to(PROJECT)), "sha256": sha(path)})
        path = run / "forecasts.csv"
        f = pd.read_csv(path).loc[lambda x: x.variant.eq("payoff_context")].copy()
        assert f.variant.unique().tolist() == ["payoff_context"]
        assert pd.to_datetime(f.decision_date).max() < pd.Timestamp("2024-01-01")
        assert pd.to_datetime(f.entry_date).max() < pd.Timestamp("2024-01-01")
        forecasts.append(f)
        source_files.append({"path": str(path.relative_to(PROJECT)), "sha256": sha(path)})
        u = pd.read_csv(run / "feature_usage.csv").loc[lambda x: x.model.eq("context")].copy()
        u.insert(0, "ticker", ticker)
        usages.append(u)
        for item in json.loads((run / "calibrators.json").read_text()):
            if item["context"]:
                calibrators.append({"ticker": ticker, **item})
        c = pd.read_csv(run / "metrics.csv").loc[lambda x: x.variant.eq("payoff_context")].copy()
        costs.append(c)
        target = DEST / "data/model_rules" / ticker
        target.mkdir(exist_ok=True)
        for path in run.glob("*_context.txt"):
            shutil.copyfile(path, target / path.name)
    pd.concat(ledgers, ignore_index=True).to_csv(DEST / "data/ledgers.csv", index=False)
    pd.concat(forecasts, ignore_index=True).to_csv(DEST / "data/forecasts.csv", index=False)
    usage = pd.concat(usages, ignore_index=True)
    usage.to_csv(DEST / "data/feature_usage.csv", index=False)
    pd.concat(costs, ignore_index=True).to_csv(DEST / "data/cost_sensitivity.csv", index=False)
    (DEST / "data/calibrators.json").write_text(json.dumps(calibrators, indent=2), encoding="utf-8")
    base = pd.read_csv(PROJECT / "data/feature_registry.csv")
    ctx = pd.read_csv(PROJECT / "data/etf_context_registry.csv")
    rows = []
    for ticker in TICKERS:
        used = set(usage.loc[usage.ticker.eq(ticker), "feature"])
        for name in sorted(used):
            match = ctx.loc[ctx.ticker.eq(ticker) & ctx.feature_name.eq(name)]
            if len(match):
                rec = match.iloc[0].to_dict()
                rec["family"] = "ETF context"
                rec["interpretation"] = rec.get("interpretation_caveat", "")
            else:
                match = base.loc[base.feature_name.eq(name)]
                assert len(match) == 1, (ticker, name)
                rec = match.iloc[0].to_dict()
                rec["interpretation"] = rec.get("corporate_action_caveat", "")
            rec["ticker"] = ticker
            rows.append(rec)
    pd.DataFrame(rows).to_csv(DEST / "data/factor_dictionary.csv", index=False)
    for rel in ["src/research.py", "src/experiments_v3.py", "src/etf_context.py",
                "config/experiments_v3.json", "config/run_config.json"]:
        src = PROJECT / rel
        shutil.copyfile(src, DEST / "reference_sources" / src.name)
    for rel in ["v3_audit.json", "v3_run_manifest.json", "v3_inference_plan.json", "v3_engineering_amendment.json"]:
        shutil.copyfile(PROJECT / "reports" / rel, DEST / "reference_sources" / rel)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": "v3.1", "source_directory": "runs/v3_1",
        "strategy": "payoff_context", "tickers": TICKERS,
        "curve_labels": ["Payoff + context", "ETF buy & hold", "S&P 500 (SPY) buy & hold"],
        "cost_bps_one_way": 5, "cash_yield": 0, "risk_free_rate": 0,
        "reserved_start": "2024-01-01", "reserved_scored": False,
        "purpose": "Focused reporting export of saved results; no new training or hypothesis selection",
        "source_files": source_files,
        "source_config_sha256": sha(PROJECT / "config/experiments_v3.json"),
        "source_engine_sha256": sha(PROJECT / "src/experiments_v3.py"),
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    packages = ["numpy", "pandas", "matplotlib", "ipython", "nbformat", "ipykernel"]
    (DEST / "requirements.txt").write_text(
        "".join(f"{name}=={version(name)}\n" for name in packages), encoding="utf-8")
    (DEST / "README.txt").write_text(
        "PAYOFF + ETF CONTEXT — FOCUSED NOTEBOOK\n\n"
        "Open payoff_context_review.ipynb. Saved outputs are already executed.\n"
        "For GitHub, upload this entire focused_payoff_context folder, preserving\n"
        "data/ and src/. To rerun, install requirements.txt in a Python 3.12\n"
        "environment, choose that kernel, and Run All from this folder.\n\n"
        "This notebook reports the latest local completed model run, v3.1.\n"
        "There is no saved v3.6 run in the source workspace. The reporting export\n"
        "does not assign a new model version. No fitting, fetching, trading or\n"
        "reserved-period scoring occurs when notebook cells are rerun.\n\n"
        "Only the combined payoff_context procedure appears as a trading model.\n"
        "Every ETF has three curves: that procedure, its own buy-and-hold, and\n"
        "SPY buy-and-hold as the S&P 500 proxy. Accounts start at NAV1 immediately\n"
        "before the first included open; first-session movement/cost is retained.\n"
        "5bps one-way turnover fees and final liquidation apply; cash earns zero.\n"
        "Every account is separate. No portfolio allocation or option positions.\n\n"
        "Wins/losses refer to completed continuous invested episodes; position\n"
        "resizing stays inside an episode. Net P&L includes all actual fees.\n"
        "Forecast horizon and realized holding duration are different measures.\n"
        "Position statistics separate target sizing from drifted daily exposure.\n"
        "Factor importance is in-sample split importance, not causal evidence.\n\n"
        "All evaluated observations precede2024. These historical results remain\n"
        "exploratory; none of the parent study's225 new comparisons passed its\n"
        "Holm5%adjustment. Vendor-adjusted synthetic units approximate total\n"
        "return; XLF's2016 in-kind distribution still needs action reconciliation.\n"
        "Macro and FinBERT are not eligible fitted factors in this saved strategy.\n\n"
        "reference_sources/ contains the training-source/config snapshots and\n"
        "parent audit provenance. It is reference material; the full historical\n"
        "training dataset and complete research environment remain in the parent\n"
        "project. The portable notebook uses only the curated local result files.\n",
        encoding="utf-8",
    )


def notebook():
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md("""# Payoff + ETF context · focused research notebook

Source model run: **v3.1** · independent ETF long/cash accounts · evaluation ends December2023.

Each chart contains exactly three curves: **Payoff + context**, **ETF buy & hold**, and **S&P500 (SPY) buy & hold**. Trading statistics, annual forecasting settings and factors appear underneath. The notebook reads portable saved results; it does not retrain models or retrieve new data.

Capital is normalized to1 before the first included session open. Costs are5bps one way on actual traded notional, including final liquidation. Cash and the Sharpe risk-free convention are zero. The ETF-specific start dates differ. No portfolio is constructed."""),
        code("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, HTML
locations = [Path.cwd(), Path.cwd() / 'focused_payoff_context',
             Path.cwd() / 'outputs/swing-tree-local/focused_payoff_context']
ROOT = next((p for p in locations if (p / 'manifest.json').exists()), Path.cwd())
assert (ROOT / 'manifest.json').exists(), 'Open this notebook from its exported folder.'
sys.path.insert(0, str(ROOT / 'src'))
from notebook_views import FocusedReport
R = FocusedReport(ROOT)
display(HTML(R.workflow()))
"""),
        md("""## How to read the statistics

A **trade episode** is one continuous spell with positive exposure. Resizing does not start a new episode. Wins, losses and win rates use completed episodes after actual fees; final forced liquidation is identified. Net episode return uses account NAV immediately before entry.

**Forecast horizon** is the selected5/10/20-session label length. **Holding duration** counts actual invested sessions. Targets are reassessed weekly and positions can persist beyond a label horizon. **Target size while LONG**, **target size across all decisions**, and **daily realized exposure** have different denominators.

Technical, relationship and ETF-context inputs use completed prices. Context factors are market-price proxies, not direct yields, earnings revisions or commodity carry. Macro and FinBERT were excluded from this fitted run because their historical availability was not qualified."""),
    ]
    for ticker in TICKERS:
        cells += [
            md(f"## {ticker}"),
            code(f"R.plot('{ticker}')"),
            code(f"display(HTML(R.details('{ticker}')))"),
        ]
    cells += [
        md("""## Interpretation and research boundaries

This focused view is an extract of an exploratory historical study, not a new validation run. The parent study compared225 new strategy/ETF combinations; none passed its approximate Holm-adjusted5%threshold. Displaying only the combined procedure does not remove that prior search.

2024 onward remains reserved. Equity and SPY accounts use the same dates within each ETF section. Adjusted prices approximate total return using fractional synthetic units and no second dividend credit. Public-data revisions, simplified constant fees, zero cash yield, and the XLF2016 in-kind distribution remain qualifications.

The current improvement questions are payoff-calibration reliability, turnover/exposure effects, and qualified direct economic factors. These are future research questions; this notebook reports the saved combined model unchanged."""),
        code("display(HTML(R.verification()))"),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "swing_tree": {"source_run": "v3.1", "variant": "payoff_context",
                       "reserved_start": "2024-01-01", "reserved_scored": False,
                       "curves_per_etf": 3, "etfs": TICKERS},
    })
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for old, new in [("December2023","December 2023"), ("S&P500","S&P 500"),
                             ("to1 ","to 1 "), ("are5bps","are 5 bps"),
                             ("selected5/10/20-session","selected 5/10/20-session"),
                             ("compared225","compared 225"), ("Holm-adjusted5%threshold","Holm-adjusted 5% threshold"),
                             ("XLF2016","XLF 2016")]:
                cell.source = cell.source.replace(old,new)
    return nb


def execute(nb):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_base = WORKSPACE / "work/report-runtime/kernels_focused"
    directory = kernel_base / "python3"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python3 focused research", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_base)]))
    try:
        client = NotebookClient(nb, km=manager, timeout=90, startup_timeout=30,
                                resources={"metadata": {"path": str(DEST)}}, allow_errors=False,
                                on_cell_executed=lambda **kw: print("Executed focused cell", kw.get("cell_index",0)+1, flush=True))
        client.execute(cwd=str(DEST))
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    nbf.validate(nb)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
    nbf.write(nb, DEST / "payoff_context_review.ipynb")
    sys.path.insert(0, str(PROJECT / "src"))
    from build_deliverables import html_from_notebook
    html_from_notebook(nb, DEST / "payoff_context_review.html")
    path = DEST / "payoff_context_review.html"
    text = path.read_text(encoding="utf-8").replace("Swing Tree · measured ETF outcomes", "Payoff + ETF context · focused review").replace("local run v2", "focused payoff/context · source v3.1")
    path.write_text(text, encoding="utf-8")
    print("FOCUSED NOTEBOOK COMPLETE", flush=True)


if __name__ == "__main__":
    prepare()
    if "--prepare-only" not in sys.argv:
        execute(notebook())
