"""Direct correlation-aware alpha comparisons under the leading long bias."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];V31=HERE.parent/'v31_growth_long_bias';V34=HERE.parent/'v34_stock_risk';V35=HERE.parent/'v35_long_bias_drawdown'
sys.path.insert(0,str(V31/'src'))
from settings import read,write,sha,SOURCE,V28
import runner,learner
from allocation import target,unit_market,annual_variance
import numpy as np,pandas as pd,datetime,argparse,time,importlib.util
ARMS=('fixed_corr_long50','retrained_corr_long50')
PAIRS=[['fixed_corr_long50','monthly_leader'],['retrained_corr_long50','monthly_leader'],['retrained_corr_long50','fixed_corr_long50']]

def protocol():
    p=HERE/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip();m=read(p)
    for n,h in m['code'].items():assert sha(HERE/'src'/n)==h,n
    for n,h in m['sources'].items():assert sha(n)==h,n
    return sha(p)

def freeze():
    p=HERE/'protocol.json'
    if p.exists():return protocol()
    files=[V31/'protocol.json',V31/'src/allocation.py',V31/'src/research_account.py',V31/'src/runner.py',V31/'account_inputs/log_growth_desired.pkl',V31/'account_inputs/growth_long50_v12_desired.pkl',V34/'protocol.json',V34/'data_manifest.json',V34/'data/factor_loadings.npy',V34/'account_inputs/fixed_corr3_neutral_desired.pkl',V34/'account_inputs/retrained_corr3_neutral_desired.pkl']
    m=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Frozen before v36 outcomes; neutral correlation results already known; direct comparison explicitly requested',arms=ARMS,pairs=PAIRS,
        objective='Test the two previously constructed correlation-neutral stock-selection portfolios with the leading +.50 long-bias allocation and shared12% risk budget, not infer this result from neutral tests.',
        design='Use unchanged v34 fixed_corr3 and retrained_corr3 neutral desired portfolios. Apply exactly v31 long-bias allocator and gate handling, original21-session schedule,next-close,repaired daily account,2bps/side,gross3 active/500,name.15,buffer.2. Long broad-stock basket has equal target net and estimated SPY beta, capped .50. Alpha contracts to share risk; no independent overlay leverage.',
        interpretation='Correlation constraints apply to the neutral stock-selection component. The long market basket is intentionally allowed its common-factor exposure; total portfolio is not PCA neutral. The retrained coefficients were fitted for neutral correlation-aware allocation in v34, not refitted under this positive bias. This isolates allocation under long bias and does not test joint long-biased training.',
        controls='Regenerate original long50 targets from v31 neutral log_growth and require numerical identity on all204 dates before accounts. Original leader and original H21 remain unchanged.',
        inference=dict(primary='2012-2023',full='2007-2023',stress=[2008,2020,2022],family=3,blocks=[20,60],replications=2000,endpoint='Annual log growth paired difference; drawdown descriptive'),
        promotion=False,limitations=['Reused history and known2022 failure','No new fitting under long-biased objective','Zero borrow/funding/cash interest; conditional source valuations remain','Additional factor neutrality at scheduled alpha targets, not hard daily total-portfolio neutrality'],
        sources={str(f):sha(f) for f in files},code={'compare.py':sha(Path(__file__))})
    write(p,m);p.with_suffix('.sha256').write_text(sha(p));return sha(p)

def prepare():
    seal=protocol();d=learner.load();ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));dates=d['dates'][ix];names=d['names'];original=pd.read_pickle(V31/'account_inputs/log_growth_desired.pkl');reference=pd.read_pickle(V31/'account_inputs/growth_long50_v12_desired.pkl');source={a:pd.read_pickle(V34/f'account_inputs/{"fixed" if a.startswith("fixed") else "retrained"}_corr3_neutral_desired.pkl') for a in ARMS};loads=np.load(V34/'data/factor_loadings.npy');results={a:[] for a in ARMS};rows=[];parity=0.;residual=0.;out=HERE/'account_inputs';out.mkdir(exist_ok=True);(HERE/'reports').mkdir(exist_ok=True)
    for row,j in enumerate(ix):
        date=dates[row];gate=d['gate'][j];mask=d['mask'][j];beta=d['beta'][j]
        with np.load(V28/f'data/covariances/{date:%Y-%m-%d}.npz') as z:c=z['centered'];indices=z['indices']
        value=original.iloc[row].to_numpy();w,info=target(value/gate if gate else np.zeros(len(names)),beta,mask,c,indices,.12,.5,gate);error=float(abs(w-reference.iloc[row].to_numpy()).max());assert error<1e-10;parity=max(parity,error)
        for a in ARMS:
            assert source[a].index.equals(dates) and list(source[a].columns)==names;value=source[a].iloc[row].to_numpy();w,info=target(value/gate if gate else np.zeros(len(names)),beta,mask,c,indices,.12,.5,gate);alpha=w-unit_market(beta,mask)*info['effective_bias'];error=float(abs(loads[j]@alpha).max());assert error<1e-9;residual=max(residual,error)
            vol=float(np.sqrt(annual_variance(w,c,indices)));assert vol<=.12+1e-9 and abs(w).sum()<=3*min(1,mask.sum()/500)+1e-9 and abs(w).max()<=.15+1e-9 and abs(w@beta-w.sum())<1e-9 and abs(w[~mask]).max(initial=0)<1e-10
            results[a].append(w);rows.append(dict(arm=a,date=str(date.date()),gate=gate,forecast_vol=vol,gross=float(abs(w).sum()),factor_alpha_residual=error,**info))
    for a,w in results.items():
        f=out/f'{a}_desired.pkl';pd.DataFrame(w,index=dates,columns=names).to_pickle(f);write(out/f'{a}.json',dict(protocol_sha256=seal,target_sha256=sha(f)))
    pd.DataFrame(rows).to_csv(HERE/'reports/target_diagnostics.csv',index=False);write(HERE/'reports/preparation_qa.json',dict(passed=True,protocol_sha256=seal,original_leader_target_parity_error=parity,original_leader_parity_dates=len(dates),maximum_alpha_factor_exposure=residual,targets=2*len(dates),no_new_training=True));print('PREPARED',parity,residual,flush=True)

def accounts(arms):
    seal=protocol();loaded=runner.inputs();out=HERE/'accounts';out.mkdir(exist_ok=True)
    for a in arms:
        done=out/f'{a}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{a}_ledger.csv.gz');continue
        f=HERE/f'account_inputs/{a}_desired.pkl';m=read(f.with_name(a+'.json'));assert m['protocol_sha256']==seal and m['target_sha256']==sha(f);audit=runner.bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));count=0
        def sink(frame):
            nonlocal count
            audit.consume(frame);count+=len(frame)
        began=time.monotonic();print('ACCOUNT_BEGIN',a,flush=True);ledger,pos,actual,decisions,status=runner.simulate(loaded,pd.read_pickle(f),bias=True,positions=True,sink=sink);assert pos.empty and status['streamed_position_rows']==count and status['arithmetic_passed'];status['noncash_exposure_audit']=audit.report()
        if audit.rows:status['issues'].extend(audit.rows);status['validated_performance']=False
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{a}_{suffix}.csv.gz',index=index)
        write(done,dict(protocol_sha256=seal,target_sha256=sha(f),ledger_sha256=sha(out/f'{a}_ledger.csv.gz'),audit=status,seconds=time.monotonic()-began));print('ACCOUNT_DONE',a,round(time.monotonic()-began,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','prepare','accounts']);p.add_argument('--arms',nargs='+',default=list(ARMS));a=p.parse_args()
    if a.stage=='freeze':print(freeze())
    elif a.stage=='prepare':prepare()
    else:accounts(a.arms)
