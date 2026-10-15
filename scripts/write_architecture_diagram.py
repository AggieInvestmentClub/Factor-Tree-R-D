"""Publish the starter's research graph without maintaining a second design.

Use --sync-docs to refresh the GitHub chart page and the diagram embedded in
the initial research protocol. This does not run any research experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from build_testing_chart_starter import EDGES, NODES, render_mermaid


DIAGRAM = render_mermaid(NODES, EDGES)
PROTOCOL_START = "<!-- detailed-workflow:start -->"
PROTOCOL_END = "<!-- detailed-workflow:end -->"

PAGE_INTRO = """Proposed research workflow, not an implemented or validated trading system.
Each of the 15 stocks follows this process independently as a long/cash
backtest; there is no portfolio allocation layer.

Branches show staged alternative experiments, not methods that must all be
combined. Feedback loops advance to later cutoffs using matured outcomes
only. Optional challengers are not prerequisites for the first baseline.
"""

CHART_GUIDE = """Each stock repeats this process independently. The branches are staged
alternative experiments, not an instruction to combine all models or sizing
weights. Build the NVDA/CART fixture and fixed-exposure accounting first; add
RF/XGBoost, richer factors/FinBERT, payoff/frontier sizing, and adaptive layers
in bounded, separately recorded stages. The original 24-recipe CART screen
does not include all those later combinations.

All model, factor, signal, sizing, calibration and update-policy comparisons
replay earlier inner chronological folds. Freeze the selected recipe before
refitting and predicting the next outer interval. The outer interval scores
the complete procedure; it is not another tuning window. A common learner
family policy applies to all stocks; different fitted stock specifications
do not imply unreviewed per-stock learner-family switches.

The efficient-frontier branch is one stock plus cash, not a 15-stock optimizer.
Its mean-variance experiment needs a declared risk preference, aligned
return/risk horizons and costs. With riskless cash, maximizing Sharpe alone
does not determine an interior stock weight. Expected-payoff sizing needs
both outcome probabilities and gain/loss magnitudes, not probability alone.

V2's selected NVDA reference was a 10-session, depth-3 technical/changepoint
CART with binary exposure. Compact multi-horizon ensembles, agreement and
dispersion are optional carryover hypotheses, not V2's selected NVDA model.
RF/XGBoost, FinBERT, payoff/frontier sizing and the expanded adaptive ladder
are new-study experiments. The HMM/state/stress branch is explicitly later
and is not a dependency of the initial baseline. No historical TLT long/short
or leverage rules are inherited. See the [preserved V2 report](../reference/evidence/v2/REPORT.md).

The feedback queue receives recorded forecasts and newly matured labels
directly. Scheduled or drift-triggered updates do not have to wait for a
completed outer report; every update must still obey the frozen policy and
its own information cutoff. Recipe reselection is a separately versioned
research step, never a rewrite of prior predictions.

This chart is a research specification, not evidence that these methods have
been implemented or will improve performance. Data retrieval, model training
and backtests are not performed by the chart generator.
"""


def write_diagram(output_path: Path) -> Path:
    """Write the design diagram and return its resolved path."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(DIAGRAM, encoding="utf-8", newline="\n")
    return output_path


def sync_documentation(repo_root: Path) -> list[Path]:
    """Refresh only the chart page and the marked protocol diagram section."""
    protocol_path = repo_root / "docs" / "initial_research_protocol.md"
    page_path = repo_root / "docs" / "my_testing_workflow.md"
    protocol = protocol_path.read_text(encoding="utf-8")
    block = (
        f"{PROTOCOL_START}\n"
        "## Detailed initial testing workflow\n\n"
        f"{CHART_GUIDE}\n"
        f"```mermaid\n{DIAGRAM}```\n"
        f"{PROTOCOL_END}"
    )

    if PROTOCOL_START in protocol or PROTOCOL_END in protocol:
        if protocol.count(PROTOCOL_START) != 1 or protocol.count(PROTOCOL_END) != 1:
            raise ValueError("Expected exactly one complete protocol diagram block")
        start = protocol.index(PROTOCOL_START)
        end = protocol.index(PROTOCOL_END)
        if end < start:
            raise ValueError("Protocol diagram markers are out of order")
        protocol = protocol[:start] + block + protocol[end + len(PROTOCOL_END):]
    else:
        anchor = "## Tonight's order"
        if protocol.count(anchor) != 1:
            raise ValueError("Cannot locate the protocol insertion point uniquely")
        protocol = protocol.replace(anchor, block + "\n\n" + anchor, 1)

    page = (
        "# Proposed initial stock research workflow\n\n"
        f"{PAGE_INTRO}\n"
        f"```mermaid\n{DIAGRAM}```\n"
    )
    protocol_path.write_text(protocol, encoding="utf-8", newline="\n")
    page_path.write_text(page, encoding="utf-8", newline="\n")
    return [protocol_path, page_path]


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "docs" / "long_only_swing_architecture.mmd"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=default)
    parser.add_argument(
        "--sync-docs", action="store_true",
        help="also refresh the GitHub page and the marked protocol diagram",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(write_diagram(args.output))
    if args.sync_docs:
        for path in sync_documentation(Path(__file__).resolve().parents[1]):
            print(path)
