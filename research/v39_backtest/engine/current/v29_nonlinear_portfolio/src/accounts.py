from common import *
import horizon_accounts as bridge
import gc,time,argparse

def contract():return dict(protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'),source_status=bridge.legacy.sources())
def inputs():
    a=bridge.legacy.inputs(need_prices=True);del a['panel'];gc.collect();return a
def simulate(loaded,targets,*,start=None,end=None,positions=False,sink=None):
    clock=loaded['clock']
    if start is None:start=clock[clock.get_loc(targets.index[0])+1]
    if end is None:end=clock[-1]
    # Source claims require settlement inside an account window. Filter records
    # with future ex-dates, but never silently discard an accrued receivable.
    claims=[r for r in loaded['claims'] if pd.Timestamp(start)<=pd.Timestamp(r['ex_date'])<=pd.Timestamp(end)]
    assert all(pd.Timestamp(r['settlement_date'])<=pd.Timestamp(end) for r in claims), 'Validation boundary crosses an unsettled legal claim'
    return bridge.account_engine.simulate_stock_portfolio(loaded['prices'],targets,loaded['events'],start=start,end=end,
        cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],neutral=True,dollar_neutral=True,beta_neutral=True,
        eligibility=loaded['eligible'],position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff='2024-01-01',
        record_positions=positions,position_sink=sink,distribution_claims=claims)

def validate_candidate(loaded,targets,end):
    requested_end=end
    # The sealed account cannot end with a legally accrued, unsettled claim.
    # End before that ex-date for every candidate in that refit window.
    crossed=[pd.Timestamp(r['ex_date']) for r in loaded['claims'] if targets.index[0]<pd.Timestamp(r['ex_date'])<=end<pd.Timestamp(r['settlement_date'])]
    if crossed:
        end=loaded['clock'][loaded['clock']<min(crossed)][-1]
        targets=targets.loc[targets.index<end]
    start=time.monotonic();ledger,_,_,decisions,audit=simulate(loaded,targets,end=end)
    r=ledger.daily_return.to_numpy();utility=252*r.mean()-2.5*252*np.var(r,ddof=1)
    assert audit['arithmetic_passed']
    return dict(utility=float(utility),annual_mean=float(252*r.mean()),annual_variance=float(252*np.var(r,ddof=1)),
        sessions=len(r),start=str(ledger.index[0]),end=str(ledger.index[-1]),provisional_sessions=audit['provisional_sessions'],
        actual_engine=True,submitted=int(decisions.submitted.sum()),seconds=time.monotonic()-start,
        requested_end=str(requested_end),unsettled_claim_boundary_shortened=bool(crossed))

def run(arms):
    c=contract();d=load();loaded=inputs();out=ROOT/'accounts';out.mkdir(exist_ok=True)
    for arm in arms:
        done=out/f'{arm}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{arm}_ledger.csv.gz');continue
        file=ROOT/f'account_inputs/{arm}_desired.pkl';targets=pd.read_pickle(file);start=time.monotonic()
        print('ACCOUNT_BEGIN',arm,flush=True)
        noncash=bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));rows=0
        def consume(frame):
            nonlocal rows
            noncash.consume(frame);rows+=len(frame)
        ledger,pos,actual,decisions,audit=simulate(loaded,targets,positions=True,sink=consume)
        assert pos.empty and rows==audit['streamed_position_rows'] and audit['arithmetic_passed']
        audit['noncash_exposure_audit']=noncash.report()
        if noncash.rows:audit['issues'].extend(noncash.rows);audit['validated_performance']=False
        parity=None
        if arm=='h21_replay':
            ref=pd.read_csv(V23/f'accounts/{CHALLENGER}_ledger.csv.gz',index_col=0,parse_dates=True)
            assert ledger.index.equals(ref.index);numeric=ledger.select_dtypes('number').columns.intersection(ref.select_dtypes('number').columns)
            parity={k:float(np.max(np.abs(ledger[k]-ref[k]))) for k in numeric};assert max(parity.values())<1e-10
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        write(done,dict(**c,arm=arm,target_sha256=sha(file),ledger_sha256=sha(out/f'{arm}_ledger.csv.gz'),audit=audit,
            baseline_parity=parity,seconds=time.monotonic()-start))
        print('ACCOUNT_DONE',arm,round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=[*MODELS,'h21_replay']);a=p.parse_args();run(a.arms)
