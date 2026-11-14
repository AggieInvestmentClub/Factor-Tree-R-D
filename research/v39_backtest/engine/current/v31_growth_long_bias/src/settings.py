"""Controlled growth/long-bias experiment on immutable v23/v29 inputs."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1];V29=ROOT.parent/'v29_nonlinear_portfolio';V23=ROOT.parent/'v23_horizon_targets';V28=ROOT.parent/'v28_direct_portfolio'
SOURCE=Path(str(Path(__file__).resolve().parents[3] / 'prior/v17_long_history'))
sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(V23/'src'));sys.path.insert(0,str(V29/'src'))
CHALLENGER='h21_projected_scaled_matched'
OBJECTIVES=('utility5','utility1','net_return','log_growth')
PENALTIES=(.1,.01);EPOCHS=100
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(o,indent=2,default=str),encoding='utf-8');t.replace(p)
def load():
    import common as old
    return old.load()
def selected(d,cut,start):return np.flatnonzero((d['dates']>=start)&(d['dates']<cut)&(d['ends']<cut)&d['complete'])
def arm_specs():
    arms={'h21_legacy':dict(source='h21',repair=False,bias=0,vol=.12),'h21_repaired':dict(source='h21',repair=True,bias=0,vol=.12),'rff_repaired':dict(source='rff',repair=True,bias=0,vol=.12)}
    for obj in OBJECTIVES:arms[obj]=dict(source=obj,repair=True,bias=0,vol=.12)
    for vol in (.16,.20):arms[f'growth_v{round(vol*100)}']=dict(source='log_growth',repair=True,bias=0,vol=vol)
    for bias in (.25,.5):
        for vol in (.12,.16):arms[f'growth_long{round(bias*100)}_v{round(vol*100)}']=dict(source='log_growth',repair=True,bias=bias,vol=vol)
        arms[f'market_long{round(bias*100)}']=dict(source='market',repair=True,bias=bias,vol=.12)
    return arms
def freeze():
    p=ROOT/'protocol.json'
    if p.exists():return protocol()
    specs=arm_specs()
    pairs=[['h21_repaired','h21_legacy'],['rff_repaired','old_rff'],['utility5','rff_repaired'],
        ['utility1','utility5'],['net_return','utility5'],['log_growth','utility5'],['growth_v16','log_growth'],['growth_v20','log_growth'],
        ['growth_long25_v12','log_growth'],['growth_long50_v12','log_growth'],['growth_long25_v16','growth_v16'],['growth_long50_v16','growth_v16'],
        ['growth_long25_v12','market_long25'],['growth_long50_v12','market_long50']]
    s=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Frozen before v31 outcomes; adaptive research on reused history',
        arms=specs,objectives=OBJECTIVES,pairs=pairs,
        repair='For stale targets with missing betas, exclude currently ineligible/unquoted/terminated/nonfinite-risk names and project remaining weights into the declared constraints. Keep next-close quote checks, abort behavior, fees, corporate cash and claims unchanged. Never invent missing risk or fills.',
        controlled_long='Long-only broad-stock basket with near-equal weights and unit estimated beta; overlay25% or50% capital net long. Bias is reduced if necessary to fit total forecast risk or caps. Alpha is contracted to fit remaining budget. Net and beta targets equal the effective overlay. Bias persists when alpha efficacy gate is off.',
        guards='Neutral beta guard.05 unchanged; biased accounts guard actual beta and actual net against latest desired net, with.05 deviation bands. All guards submit next-close orders, not immediate fills. Reproject stale biased targets to current eligible names while maintaining desired net/beta.',
        constraints='21observed-session cadence;original58causal price features and32fixed random features;gross3 adjusted active/500;name.15;buffer.2;grossguard3.3;nextclose;2bps/side;no new data or post2023 outcomes.',
        objectives_description={'utility5':'annual mean net return minus2.5 annual variance','utility1':'annual mean net return minus.5 annual variance','net_return':'annual mean net return under fixed hard exposure/risk limits','log_growth':'annual mean log(1+net return), monotone in compounded growth'},
        fitting=dict(epochs=EPOCHS,penalties=PENALTIES,architecture='Same v29 random map and initialization, fixed seed20261111',
            validation='Prior10year annual refits; common mature monthly census; last2years validation; strict inner and outer endpoint purge; select regularization with repaired daily engine and objective matching the arm. Fixed.1 fallback2007/2008. Same shortened2012 validation claim boundary.',
            gradient='v29 sequential21session surrogate retained, objective varied only. Daily repaired engine used for model selection and final accounts. Measure remaining surrogate gap; no claim of differentiation through daily guards.'),
        inference=dict(primary='2012-2023',early='2007-2011',full='2007-2023',blocks=[20,60],replications=2000,holm_family=len(pairs),primary_endpoint='annual log growth difference; CAGR monotone transformation',secondary='Sharpe,drawdown,realized risk,market beta,fees,exposure'),
        funding='Original accounting assumes zero borrow,financing,cash yield. Add clearly labeled100/300bp short-borrow sensitivities without claiming actual borrow availability or historical prices.',
        promotion=False,
        sources={str(p):sha(p) for p in [SOURCE/'src/account_engine.py',V29/'protocol.json',V29/'data_manifest.json',V29/'src/policy.py',V29/'src/common.py',V23/'account_contract.json',V23/f'accounts/{CHALLENGER}_complete.json']},
        code={p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py')) if p.name not in ('evaluate.py','report.py','qa.py')},
        limitations=['Original conditional source valuations and historical universe retained','No statistical correction for earlier research families','A long bias can increase CAGR through market exposure rather than better alpha','Gradient replay remains approximate; high estimated training growth is not a forecast'])
    write(p,s);p.with_suffix('.sha256').write_text(sha(p));return sha(p)
def protocol():
    p=ROOT/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip()
    for n,h in read(p)['code'].items():assert sha(ROOT/'src'/n)==h,n
    return sha(p)
