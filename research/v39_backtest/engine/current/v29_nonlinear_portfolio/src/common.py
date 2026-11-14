"""v29: execution-aware nonlinear portfolio research; source accounts immutable."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
V28=ROOT.parent/'v28_direct_portfolio'
V23=ROOT.parent/'v23_horizon_targets'
SOURCE=Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(V23/'src'))
MODELS=('replay_linear','replay_random','replay_mlp')
CHALLENGER='h21_projected_scaled_matched'
PENALTIES=(.1,.01)
EPOCHS=100
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(o,indent=2,default=str),encoding='utf-8');q.replace(p)
def freeze():
    p=ROOT/'protocol.json'
    if p.exists():return protocol()
    spec=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      status='Frozen before v29 fits and outcomes; adaptive research on reused historical data',
      benchmark=CHALLENGER,models=MODELS,penalties=PENALTIES,epochs=EPOCHS,seed=20261111,
      features='Same36 price/sector/availability inputs; daily causal EWMA .25, plus slow half-life21 bank for22 continuous features only. No company data added to the core comparison.',
      architecture=dict(linear='58 shared coefficients',random='58 linear plus32 fixed cos random features, seed20261111, frequencies N(0,1/58), phases uniform0..2pi',mlp='58-input8-tanh-hidden shared network with linear skip'),
      objective=dict(training='Sequential21-session surrogate: current-dollar holdings drift, endogenous NAV, implicit post-fee rebalance, hardL1.2 buffer and scheduled beta/gross guard, common known gate, exact decision-time source covariance/risk and neutrality/caps. Annual mean net return minus2.5*annual ex-ante variance minus ridge.',
        risk_target=.12,gamma=5,cost_per_side=.0002,buffer=.2,gross_guard=3.3,beta_guard=.05,
        timing='Rewards are source next-close-to-next-close21-session returns. Surrogate evaluates buffer on execution-boundary drifted holdings, one close later than actual engine.',
        persistence='Fast and slow causal characteristic histories plus holdings/cost state. No stock-specific parameters in core arms.',
        gaps='Common complete-query census; missing reward or nonconsecutive dates reset to cash and charge liquidation of previous segment. No zero-return substitution for known assets.',
        terminal='Surrogate charges liquidation at every segment end; account selection liquidates at validation end as source engine does.',
        pre2007_gate='Gate unavailable before original OOS begins; multiplier1 used in training before2007, not represented as historical OOS gate.',
        omissions=['No daily intraperiod guards or margin calls in gradient surrogate','Endpoint total returns do not reproduce timing of corporate cash/claims or quote failures','Buffer timing differs by one close; current risk-ineligible holdings force surrogate rebalance even when the original buffer might retain them','Hard branches differentiated piecewise; nonconvex Adam fits have no global-optimum certificate']),
      validation='Annual prior10year rolling window; complete H21 endpoints strictly before cutoff; last2calendar years withheld and inner endpoints strictly before validation start. For each penalty fit100epochs, select by original daily account engine annual mean net return minus2.5 annual realized variance on validation, including actual gate/buffer/guards/corporate events. Before2009 or insufficient inner queries use .1 fixed. Refit selected penalty on full mature sample.',
      unchanged_execution='Original engine, next close,21 observed-session cadence, shared originalH5 gate,12% risk target,gross3 adjusted active/500,name.15,dollar/SPYbeta neutrality,buffer.2,2bps/side,no borrow/financing.',
      pairs=[[m,CHALLENGER] for m in MODELS]+[['replay_random','replay_linear'],['replay_mlp','replay_linear']],
      inference=dict(primary='2012-2023',early='2007-2011',full='2007-2023',blocks=[20,60],replications=2000,holm_family=5,promotion=False),
      sources={str(p):sha(p) for p in [V28/'data_manifest.json',V23/'protocol.json',V23/f'accounts/{CHALLENGER}_complete.json',V23/'account_inputs/original_huber_gate_cadence21.csv',SOURCE/'data/candidate_features/panel.pkl',V23/'data/horizon_labels.pkl']},
      code={p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py')) if p.name not in ('report.py','evaluate.py')},
      limits=['Modern source reconstruction, conditional historical universe and unresolved valuation sessions remain','Same underlying data/constraints, with changed economic objective and functional form; no fresh untouched test set','Only small fixed nonlinear architectures; no assertion that all neural or nonlinear methods fail'])
    write(p,spec);p.with_suffix('.sha256').write_text(sha(p));return sha(p)
def protocol():
    p=ROOT/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip()
    s=read(p)
    for name,h in s['code'].items():assert sha(ROOT/'src'/name)==h,name
    return sha(p)
def load():
    m=read(ROOT/'data/metadata.json');d={n:np.load(ROOT/f'data/{n}.npy',mmap_mode='r') for n in ('x','centered','diag','orth','mask','beta','returns','complete','gate')}
    d.update(m);d['dates']=pd.DatetimeIndex(m['dates']);d['ends']=pd.DatetimeIndex(m['label_ends']);return d
def selected(d,cut,start):return np.flatnonzero((d['dates']>=start)&(d['dates']<cut)&(d['ends']<cut)&d['complete'])
