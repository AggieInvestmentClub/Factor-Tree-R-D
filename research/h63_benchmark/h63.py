"""Portable H63 replay from an authorized, external, hash-verified input pack."""
from pathlib import Path
import argparse,hashlib,json,os,sys,types
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str,allow_nan=False)+'\n')
def external(p):
    p=Path(p).expanduser().resolve();repo=ROOT.parents[1]
    if p==repo or repo in p.parents or p in repo.parents:raise ValueError('Use a directory outside this Git checkout')
    return p
def verify(inputs):
    import ast
    manifest=json.loads((ROOT/'data_contract/input_manifest.json').read_text())
    for row in manifest['files']:
        f=inputs/row['path'];assert inputs in f.resolve().parents
        assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],row['path']
    for f in ROOT.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
    package=ROOT/'provenance/package_manifest.json'
    if package.exists():
        for row in json.loads(package.read_text())['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    return dict(passed=True,files=len(manifest['files']),private_bytes=sum(r['bytes'] for r in manifest['files']))
def replay(inputs,out,part):
    import numpy as np,pandas as pd
    import account,account_cache
    from model import limits
    proxy=types.SimpleNamespace(exits=types.SimpleNamespace(engine=lambda:account));engine,_=account_cache.build(proxy)
    prices=pd.read_pickle(inputs/'account/prices.pkl');normalization=0
    for frame in prices.values():
        missing=frame.distribution_reinvestment_turnover.isna();assert frame.loc[missing,'q'].ne(1).all()
        normalization+=int(missing.sum());frame.loc[missing,'distribution_reinvestment_turnover']=0.
    start,end,cutoff=('2007-01-04','2023-12-29','2024-01-01') if part=='historical' else ('2024-01-02','2026-11-11','2026-11-12')
    desired=pd.read_pickle(out/f'{part}_desired.pkl')
    claims=[r for r in json.loads((inputs/'account/claims.json').read_text()) if start<=r['ex_date']<=end]
    def sink(frame):pass
    def arrays(date,shares,values,valid,provisional,broken):assert shares.min()>=-1e-12
    sink.consume_arrays=arrays
    ledger,_,targets,decisions,audit=engine.simulate_stock_portfolio(prices,desired,pd.read_csv(inputs/'account/events.csv',parse_dates=['effective_date']),start=start,end=end,cost_bps=2,buffer_l1=.2,risk_betas=pd.read_pickle(inputs/'account/beta.pkl'),eligibility=pd.read_pickle(inputs/'account/eligible.pkl'),neutral=False,dollar_neutral=False,beta_neutral=False,controlled_long_bias=False,repair_stale_risk=True,position_cap=.15,gross_cap=1.,gross_guard=1.1,data_cutoff=cutoff,record_positions=True,position_sink=sink,position_batch_rows=1,distribution_claims=claims,directional_limits=limits())
    ref=pd.read_csv(ROOT/f'reference/{part}_ledger.csv.gz',index_col=0,parse_dates=True)
    assert ledger.index.equals(ref.index);errors={}
    for c in ledger.select_dtypes(include=['number','bool']).columns.intersection(ref.columns):
        a=ledger[c].to_numpy(dtype=float);b=ref[c].to_numpy(dtype=float);np.testing.assert_allclose(a,b,atol=1e-9,rtol=0,equal_nan=True)
        errors[c]=float(np.nanmax(abs(a-b))) if np.isfinite(a-b).any() else 0.
    assert audit['arithmetic_passed'] and ledger.cash.min()>=0
    assert (pd.to_datetime(decisions.execution_date)>pd.to_datetime(decisions.decision_date)).all()
    ledger.to_csv(out/f'{part}_ledger.csv.gz');decisions.to_csv(out/f'{part}_decisions.csv.gz',index=False)
    write(out/f'{part}_audit.json',audit)
    return dict(passed=True,part=part,sessions=len(ledger),max_ledger_error=max(errors.values()),numeric_field_errors=errors,action_metadata_empty_rows=normalization,price_or_return_imputed=False,provisional_sessions=audit['provisional_sessions'])
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','targets','replay','fit-ranker','refit-capital']);p.add_argument('--inputs',required=True);p.add_argument('--output');p.add_argument('--part',choices=['historical','forward']);p.add_argument('--year',type=int);a=p.parse_args()
    inputs=external(a.inputs)
    if a.command=='verify':result=verify(inputs)
    else:
        if not a.output:p.error('--output required')
        out=external(a.output);out.mkdir(parents=True,exist_ok=True)
        from model import targets,fit_ranker
        if a.command=='targets':result=targets(inputs,out)
        elif a.command=='replay':
            if not a.part:p.error('--part required')
            result=replay(inputs,out,a.part)
        elif a.command=='fit-ranker':
            if a.year not in range(2007,2027):p.error('--year must be 2007..2026')
            result=fit_ranker(inputs,a.year,out)
        else:
            if a.year not in range(2007,2027):p.error('--year must be 2007..2026')
            import numpy as np,pandas as pd,torch
            from model import load
            from capital_training import train
            d=load(inputs);cut=pd.Timestamp(a.year,1,1);ix=np.flatnonzero(d['complete']&(d['dates']<cut)&(d['dates']>=cut-pd.DateOffset(years=10))&(d['ends']<cut))
            meta=json.loads((ROOT/f'checkpoints/capital/{a.year}.json').read_text());assert len(ix)==meta['training_periods']
            model,stats=train(d,ix,'long_only',meta['selected_penalty']);ref=torch.load(ROOT/f'checkpoints/capital/{a.year}.pt',weights_only=True)
            error=max(float((v-ref[k]).abs().max()) for k,v in model.state_dict().items());assert error<1e-10,error
            torch.save(model.state_dict(),out/f'capital_{a.year}.pt');result=dict(passed=True,year=a.year,max_parameter_error=error,selected_penalty=meta['selected_penalty'],training=stats,selection_repeated=False)
        write(out/(a.command+('_'+a.part if a.part else '')+('_'+str(a.year) if a.year else '')+'_qa.json'),result)
    print(json.dumps(result,default=str),flush=True)
if __name__=='__main__':main()
