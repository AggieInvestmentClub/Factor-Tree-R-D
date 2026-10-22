"""Read-only checks of the tracked report; no fetching, fitting or reserved scores."""
from pathlib import Path
import json
import sys
import nbformat
project=Path(__file__).resolve().parents[1]
root=project/"focused_payoff_context"
sys.path.insert(0,str(root/"src"))
from diagnostic_audit import audit_bundle
result=audit_bundle(root,require_extended=True,check_manifest=True)
assert result["status"]=="PASS",result["violations"]
nb=nbformat.read(root/"payoff_context_review.ipynb",as_version=4)
nbformat.validate(nb)
cells=[c for c in nb.cells if c.cell_type=="code"]
assert len(cells)==139 and all(c.execution_count is not None for c in cells)
outputs=[o for c in cells for o in c.outputs]
assert not any(o.output_type=="error" for o in outputs)
images=sum("image/png" in o.get("data",{}) for o in outputs)
assert images==61
assert nb.metadata.swing_tree.variant=="payoff_context"
assert nb.metadata.swing_tree.source_run=="v3.1"
assert nb.metadata.swing_tree.reserved_start=="2024-01-01"
print(json.dumps({"status":"PASS","diagnostic_checks":result["checks_passed"],
                  "executed_cells":len(cells),"embedded_figures":images,
                  "reserved_scored":False},indent=2))
