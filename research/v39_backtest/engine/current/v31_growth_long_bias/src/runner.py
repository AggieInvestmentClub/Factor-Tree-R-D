from settings import *
import importlib.util,time,gc,argparse
import horizon_accounts as bridge
# Explicit path avoids the original engine already loaded by the immutable loader.
spec=importlib.util.spec_from_file_location('v31_research_account',ROOT/'src/research_account.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def inputs():
    a=bridge.legacy.inputs(need_prices=True);del a['panel'];gc.collect();return a
def simulate(loaded,targets,*,repair=True,bias=False,end=None,positions=False,sink=None):
    clock=loaded['clock'];start=clock[clock.get_loc(targets.index[0])+1];end=clock[-1] if end is None else end
    claims=[r for r in loaded['claims'] if start<=pd.Timestamp(r['ex_date'])<=end]
    assert all(pd.Timestamp(r['settlement_date'])<=end for r in claims)
    return engine.simulate_stock_portfolio(loaded['prices'],targets,loaded['events'],start=start,end=end,cost_bps=2,buffer_l1=.2,
        risk_betas=loaded['beta'],neutral=not bias,dollar_neutral=not bias,beta_neutral=not bias,eligibility=loaded['eligible'],
        position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff='2024-01-01',record_positions=positions,position_sink=sink,
        distribution_claims=claims,repair_stale_risk=repair,controlled_long_bias=bias)
def validation(loaded,target,end,objective):
    crossed=[pd.Timestamp(r['ex_date']) for r in loaded['claims'] if target.index[0]<pd.Timestamp(r['ex_date'])<=end<pd.Timestamp(r['settlement_date'])]
    if crossed:end=loaded['clock'][loaded['clock']<min(crossed)][-1];target=target.loc[target.index<end]
    t=time.monotonic();ledger,_,_,_,audit=simulate(loaded,target,end=end);r=ledger.daily_return.to_numpy()
    mu=252*r.mean();var=252*r.var(ddof=1);growth=252*np.log1p(r).mean()
    utility={'utility5':mu-2.5*var,'utility1':mu-.5*var,'net_return':mu,'log_growth':growth}[objective]
    assert audit['arithmetic_passed']
    return dict(utility=float(utility),annual_mean=float(mu),annual_log_growth=float(growth),annual_variance=float(var),start=str(ledger.index[0]),end=str(ledger.index[-1]),sessions=len(r),claim_boundary_shortened=bool(crossed),seconds=time.monotonic()-t,exact_repaired_engine=True)
def run(arms):
    seal=protocol();loaded=inputs();out=ROOT/'accounts';out.mkdir(exist_ok=True);specs=arm_specs()
    for name in arms:
        done=out/f'{name}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{name}_ledger.csv.gz');continue
        a=specs[name];file=ROOT/f'account_inputs/{name}_desired.pkl';targets=pd.read_pickle(file);prepared=read(ROOT/f'account_inputs/{name}.json')
        assert sha(file)==prepared['target_sha256'] and prepared['protocol_sha256']==seal
        started=time.monotonic();print('ACCOUNT_BEGIN',name,flush=True)
        noncash=bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));count=0
        def sink(frame):
            nonlocal count
            noncash.consume(frame);count+=len(frame)
        ledger,pos,actual,decisions,audit=simulate(loaded,targets,repair=a['repair'],bias=a['bias']>0,positions=True,sink=sink)
        assert pos.empty and audit['streamed_position_rows']==count and audit['arithmetic_passed'];audit['noncash_exposure_audit']=noncash.report()
        if noncash.rows:audit['issues'].extend(noncash.rows);audit['validated_performance']=False
        parity=None
        if name=='h21_legacy':
            ref=pd.read_csv(V23/f'accounts/{CHALLENGER}_ledger.csv.gz',index_col=0,parse_dates=True);assert ledger.index.equals(ref.index)
            cols=ledger.select_dtypes('number').columns.intersection(ref.select_dtypes('number').columns);parity={c:float(np.nanmax(abs(ledger[c]-ref[c]))) for c in cols};assert max(parity.values())<1e-10
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{name}_{suffix}.csv.gz',index=index)
        write(done,dict(protocol_sha256=seal,arm=name,settings=a,target_sha256=sha(file),prepared_sha256=sha(ROOT/f'account_inputs/{name}.json'),ledger_sha256=sha(out/f'{name}_ledger.csv.gz'),engine_sha256=sha(ROOT/'src/research_account.py'),audit=audit,benchmark_parity=parity,seconds=time.monotonic()-started))
        print('ACCOUNT_DONE',name,round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=list(arm_specs()));a=p.parse_args();run(a.arms)
