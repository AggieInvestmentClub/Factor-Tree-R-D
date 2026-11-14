"""Close-triggered position exits on the frozen correlation-aware long portfolio."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];V31=HERE.parent/'v31_growth_long_bias';V33=HERE.parent/'v33_adaptive_beta';V34=HERE.parent/'v34_stock_risk';V36=HERE.parent/'v36_correlation_long_bias';V23=HERE.parent/'v23_horizon_targets'
sys.path.insert(0,str(V31/'src'))
from settings import read,write,sha,SOURCE
from allocation import unit_market,annual_variance,cap_alpha
import runner
import numpy as np,pandas as pd,datetime,time,argparse,importlib.util,difflib,pickle
ARMS=('none','fixed5','trailing8','bracket5_10','vol2sigma','tree')
PAIRS=[[a,'none'] for a in ARMS if a!='none']
FEATURES=['side','pnl','trail','age','entry_sigma21','sigma21','target_abs']

def engine_build():
    source=V31/'src/research_account.py';text=source.read_text();old='repair_stale_risk=False,controlled_long_bias=False):';assert text.count(old)==1;text=text.replace(old,'repair_stale_risk=False,controlled_long_bias=False,position_exit_policy=None):')
    old="        distance=float(np.abs(base-actual).sum());reason=''";assert text.count(old)==1
    new="""        exit_detail=None
        if position_exit_policy is not None:
            quote_mask=quotes.row('__close_ok',date)&~terminated
            if declared is not None:quote_mask &= declared
            quote_mask &= np.isfinite(b) if b is not None else False
            value,exit_detail=position_exit_policy(pd.Timestamp(date),actual.copy(),base.copy(),np.array(b,copy=True),quote_mask.copy(),initial)
            base=np.asarray(value,dtype=float)
            if base.shape!=(n,) or not np.isfinite(base).all() or abs(base[~quote_mask]).max(initial=0)>1e-10:raise ValueError('Invalid exit-policy target')
            if abs(base).max()>position_cap+1e-9 or abs(base).sum()>gross_cap*min(1.,quote_mask.sum()/STOCK_SLOTS)+1e-9:raise ValueError('Exit policy exceeds exposure limits')
            reference_beta=float(base.sum())
            if reference_beta < -1e-9 or reference_beta>.5+1e-9 or abs(base[quote_mask]@b[quote_mask]-reference_beta)>1e-9:raise ValueError('Exit policy violates controlled bias')
            mandatory=float(np.abs(actual).sum())>gross_guard+1e-12 or (abs(abeta-reference_beta)>beta_guard+1e-12 if abeta is not None else exposed.any()) or abs(float(actual.sum())-reference_beta)>beta_guard+1e-12 or bool(exit_detail.get('force_exit',False))
        distance=float(np.abs(base-actual).sum());reason=''"""
    text=text.replace(old,new);old='        if target_callback is not None:\n            decision_rows[-1].update(target_callback_used=True';assert text.count(old)==1;text=text.replace(old,"        if position_exit_policy is not None:\n            decision_rows[-1].update(position_exit_diagnostics_json=json.dumps(exit_detail,allow_nan=False))\n"+old)
    out=HERE/'src/exit_account.py';out.write_text(text);(HERE/'engine_changes.diff').write_text(''.join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),fromfile='v31_account.py',tofile='v38_exit_account.py')))

def engine():
    s=importlib.util.spec_from_file_location('v38_engine',HERE/'src/exit_account.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def protocol():
    p=HERE/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip();m=read(p)
    for n,h in m['code'].items():assert sha(HERE/'src'/n)==h,n
    for n,h in m['sources'].items():assert sha(n)==h,n
    return sha(p)

def freeze():
    p=HERE/'protocol.json'
    if p.exists():return protocol()
    files=[V31/'src/research_account.py',V31/'src/allocation.py',V36/'protocol.json',V36/'account_inputs/fixed_corr_long50_desired.pkl',V36/'accounts/fixed_corr_long50_ledger.csv.gz',V33/'account_inputs/qualified_returns.npy',V33/'account_inputs/return_clock.json',V34/'data/factor_loadings.npy',V34/'data_manifest.json',V23/'data/horizon_labels.pkl',V23/'data/label_manifest.json']
    files=[f for f in files if f.exists()]
    m=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Frozen before stop outcomes; prior2022 drawdowns known; independent of running beta-cap results',arms=ARMS,pairs=PAIRS,benchmark='Immutable v36 fixed_corr_long50, original21-session forecasts and +.50 maximum beta. No new entry alpha or beta cap selection.',
        stops={'none':'Original account replay with no policy; collect causal position states and require exact ledger identity.','fixed5':'Exit after adjusted close price loses5% versus average entry basis.','trailing8':'Exit after8% adverse movement from favorable adjusted closing-price extreme since entry; long high-water/short low-water.','bracket5_10':'Fixed5% loss or10% gain versus entry.','vol2sigma':'Entry-frozen loss threshold2*63-session daily stock volatility*sqrt21, clipped3%-20%.','tree':'Annual depth2 continuation-value regression tree trained on prior5years of every-fifth-session baseline position states with matureH21 labels; min_leaf1000,min_training10000;before2012 use fixed5. Exit if predicted holding advantage below minus20bp and age>=5sessions. No grid search.'},
        execution='Signals observed at close; orders at next close with actual quotes,fees and corporate mechanics. These are not resting intraday stop orders and thresholds do not guarantee fill prices. Reliable high/low is not available on all qualified OC dates; intraday backtest deferred rather than invent missing paths.',
        position_state='Actual executed shares from daily position sink before decision. Average adjusted entry basis updates on same-side additions, unchanged on reductions; resets on flat/sign reversal. Closing high/low watermark updates on valid marks only. Stop percentages use adjusted total-return wealth prices, not raw broker quotes. Provisional marks do not trigger exits.',
        reentry='Triggered names excluded until next original21-session forecast; persistent blocked holdings force a retry after failed execution. A fresh forecast permits re-entry with a new actual entry basis if previously flat. Triggered exits bypass20% trading buffer.',
        portfolio='After exclusions, project surviving source target jointly to dollar/beta/PCA neutrality for its alpha, then add feasible long basket over remaining names. Net/beta request is remaining raw net clipped0-.50. Gross budget never exceeds pre-projection surviving target gross or remaining-cohort cap. No normalization back to original leverage. Existing12% forecast-risk/name/gross limits remain. Guard rebalancing can change other names to restore constraints; this is an integrated portfolio exit test, not an isolated ticket-profit sum.',
        tree_label='Use audited next-close t+1 to t+22 total-wealth H21 labels including resolved corporate outcomes. Holding advantage versus exiting at t+1 is approximately(side-2bp)*future_return; same exit-cost constant cancels. This is a continuation-value proxy, not full optimal-stopping Bellman recursion. Baseline-state training is off-policy relative to the exit strategy; no claim of globally optimal exit zones. Strict label_end<annualcut, training dates beforecut, past5years; no threshold selection on2022.',
        inference=dict(primary='2012-2023',full='2007-2023',stress=[2008,2020,2022],family=5,blocks=[20,60],replications=2000,endpoint='Paired annual log growth; drawdown descriptive'),
        funding='Same2bps per side,zero borrow/funding/cash interest and conditional source valuations',promotion=False,sources={str(f):sha(f) for f in files},code={f.name:sha(f) for f in (HERE/'src').glob('*.py') if f.name in ('exits.py','exit_account.py','test_exits.py')})
    write(p,m);p.with_suffix('.sha256').write_text(sha(p));return sha(p)

def trigger(rule,pnl,trail,entry_sigma,age,prediction=None):
    if rule=='fixed5':return pnl<=-.05
    if rule=='trailing8':return trail>=.08
    if rule=='bracket5_10':return (pnl<=-.05)|(pnl>=.10)
    if rule=='vol2sigma':return pnl<=-np.clip(2*entry_sigma,.03,.20)
    if rule=='tree':return (prediction<-.002)&(age>=5)
    return np.zeros(len(pnl),bool)

class Policy:
    def __init__(self,rule,loaded,desired,collect=False):
        self.rule=rule;self.names=list(desired.columns);self.desired=desired;self.loaded=loaded;self.collect=collect;meta=read(V33/'account_inputs/return_clock.json');assert meta['names']==self.names;self.clock=pd.DatetimeIndex(meta['dates']);self.returns=np.load(V33/'account_inputs/qualified_returns.npy',mmap_mode='r');sm=read(V34/'data_manifest.json');self.sdates=pd.DatetimeIndex(sm['dates']);self.loads=np.load(V34/'data/factor_loadings.npy');n=len(self.names)
        self.entry=np.full(n,np.nan);self.peak=np.full(n,np.nan);self.shares=np.zeros(n);self.side=np.zeros(n);self.age=np.zeros(n);self.entrysigma=np.zeros(n);self.blocked=np.zeros(n,bool);self.current=None;self.valid=np.zeros(n,bool);self.states=[];self.triggers=[];self.models={};self.last_new=None;self.name_index={n:i for i,n in enumerate(self.names)};self.seal=(HERE/'protocol.sha256').read_text().strip()
    def consume(self,frame):
        assert frame.Date.nunique()==1;date=pd.Timestamp(frame.Date.iloc[0])
        if date>self.clock[-1]:return
        f=frame.set_index('ticker').reindex(self.names);shares=f.closing_shares.to_numpy();side=np.sign(shares);price=np.divide(f.closing_value.to_numpy(),shares,out=np.full(len(shares),np.nan),where=abs(shares)>1e-14);valid=f.close_quote_valid.to_numpy(bool)&~f.mark_provisional.to_numpy(bool)&~f.segment_break.to_numpy(bool)&np.isfinite(price)&(price>0)
        end=self.clock.get_loc(date)+1;x=self.returns[end-63:end];sigma=np.sqrt(np.var(x,axis=0,ddof=1)*21);same=(side==self.side)&(side!=0);fresh=(side!=0)&~same&valid;added=same&valid&(abs(shares)>abs(self.shares)+1e-12);flat=side==0
        self.entry[fresh]=price[fresh];self.peak[fresh]=price[fresh];self.age[fresh]=0;self.entrysigma[fresh]=sigma[fresh]
        add=abs(shares[added])-abs(self.shares[added]);self.entry[added]=(abs(self.shares[added])*self.entry[added]+add*price[added])/abs(shares[added]);self.age[same]+=1
        longs=valid&(side>0);shorts=valid&(side<0);self.peak[longs]=np.fmax(self.peak[longs],price[longs]);self.peak[shorts]=np.fmin(self.peak[shorts],price[shorts]);self.entry[flat]=np.nan;self.peak[flat]=np.nan;self.age[flat]=0;self.entrysigma[flat]=0
        pnl=side*(price/self.entry-1);trail=np.where(side>0,1-price/self.peak,price/self.peak-1);src=self.desired.iloc[self.desired.index.searchsorted(date,side='right')-1].to_numpy();features=np.column_stack([side,pnl,trail,np.minimum(self.age,252),self.entrysigma,sigma,abs(src)]);active=valid&(side!=0)&np.isfinite(features).all(1)
        self.current=features;self.valid=active;self.shares=shares;self.side=side;self.date=date
        if self.collect and self.clock.get_loc(date)%5==0:
            ix=np.flatnonzero(active);z=pd.DataFrame(features[ix],columns=FEATURES);z['decision_date']=date;z['ticker']=np.array(self.names)[ix];self.states.append(z)
    def __call__(self,date,actual,base,beta,quote_mask,initial):
        if date in self.desired.index and date!=self.last_new:self.blocked[:]=False;self.last_new=date
        hit=np.zeros(len(self.names),bool)
        if not initial and self.current is not None:
            assert self.date==date;pred=np.zeros(len(self.names));rule=self.rule
            if rule=='tree':
                if date.year<2012:rule='fixed5'
                else:
                    if date.year not in self.models:
                        f=HERE/f'models/tree_{date.year}.pkl';m=read(f.with_suffix('.json'));assert m['protocol_sha256']==self.seal and sha(f)==m['model_sha256'];self.models[date.year]=pickle.loads(f.read_bytes())
                    ix=np.flatnonzero(self.valid);pred[ix]=self.models[date.year].predict(self.current[ix])
            hit=trigger(rule,self.current[:,1],self.current[:,2],self.current[:,4],self.current[:,3],pred)&self.valid&~self.blocked
            for j in np.flatnonzero(hit):self.triggers.append(dict(date=str(date.date()),ticker=self.names[j],rule=rule,side=int(self.side[j]),pnl=float(self.current[j,1]),trail=float(self.current[j,2]),age=int(self.age[j]),predicted_hold_advantage=float(pred[j]) if self.rule=='tree' and rule=='tree' else None))
            self.blocked|=hit
        force=bool((self.blocked&(abs(actual)>1e-10)).any());detail=dict(new_exits=int(hit.sum()),blocked=int(self.blocked.sum()),force_exit=force)
        if not self.blocked.any():return base,detail
        end=self.clock.get_loc(date)+1;x=self.returns[end-63:end];valid=np.isfinite(x).all(0);indices=np.flatnonzero(valid);c=x[:,indices]-x[:,indices].mean(0);mask=quote_mask&~self.blocked&valid;surviving=np.where(mask,base,0.);gross=min(float(abs(surviving).sum()),3*min(1,mask.sum()/500));delta=min(.5,max(0.,float(surviving.sum())),gross)
        if mask.sum()<6 or gross<1e-12 or not np.min(beta[mask])<=1<=np.max(beta[mask]):return np.zeros(len(base)),detail
        unit=unit_market(beta,mask);vu=annual_variance(unit,c,indices);delta=min(delta,.12/np.sqrt(vu),.15/abs(unit).max());basket=delta*unit;si=self.sdates.searchsorted(date,side='right')-1;ix=np.flatnonzero(mask);matrix=np.vstack([np.ones(len(ix)),beta[ix],self.loads[si][:,ix]]);_,s,u=np.linalg.svd(matrix,full_matrices=False);rank=np.count_nonzero(s>max(matrix.shape)*np.finfo(float).eps*s[0]);u=u[:rank];a=surviving[ix]-basket[ix]
        for _ in range(2):a-=u.T@(u@a)
        alpha=np.zeros(len(base));alpha[ix]=a;va=annual_variance(alpha,c,indices);vb=annual_variance(basket,c,indices);cov=(annual_variance(alpha+basket,c,indices)-va-vb)/2;scale=1. if va<1e-20 else min(1.,max(0.,(-cov+np.sqrt(max(0.,cov*cov+va*(.12**2-vb))))/va));scale=min(scale,cap_alpha(basket,alpha,gross));w=basket+scale*alpha
        assert abs(w[self.blocked]).max(initial=0)<1e-10 and annual_variance(w,c,indices)<=.12**2+1e-9 and abs(w).sum()<=gross+1e-9;detail.update(gross_budget=gross,projected_gross=float(abs(w).sum()),target_beta=float(delta));return w,detail

def accounts(arms):
    seal=protocol();loaded=runner.inputs();desired=pd.read_pickle(V36/'account_inputs/fixed_corr_long50_desired.pkl');out=HERE/'accounts';out.mkdir(exist_ok=True);(HERE/'reports').mkdir(exist_ok=True);clock=loaded['clock'];start=clock[clock.get_loc(desired.index[0])+1];end=clock[-1];claims=[r for r in loaded['claims'] if start<=pd.Timestamp(r['ex_date'])<=end];assert all(pd.Timestamp(r['settlement_date'])<=end for r in claims)
    for arm in arms:
        done=out/f'{arm}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{arm}_ledger.csv.gz');continue
        policy=Policy(arm,loaded,desired,arm=='none');audit=runner.bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));count=0
        def sink(frame):
            nonlocal count
            audit.consume(frame);count+=len(frame);policy.consume(frame)
        began=time.monotonic();print('ACCOUNT_BEGIN',arm,flush=True);ledger,pos,actual,decisions,status=engine().simulate_stock_portfolio(loaded['prices'],desired,loaded['events'],start=start,end=end,cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],eligibility=loaded['eligible'],neutral=False,dollar_neutral=False,beta_neutral=False,controlled_long_bias=True,repair_stale_risk=True,position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff='2024-01-01',record_positions=True,position_sink=sink,position_batch_rows=1,distribution_claims=claims,position_exit_policy=None if arm=='none' else policy)
        assert pos.empty and status['streamed_position_rows']==count and status['arithmetic_passed'];status['noncash_exposure_audit']=audit.report()
        if audit.rows:status['issues'].extend(audit.rows);status['validated_performance']=False
        parity=None
        if arm=='none':
            ref=pd.read_csv(V36/'accounts/fixed_corr_long50_ledger.csv.gz',index_col=0,parse_dates=True);cols=ledger.select_dtypes('number').columns.intersection(ref.select_dtypes('number').columns);parity=float(np.nanmax(abs(ledger[cols].to_numpy()-ref[cols].to_numpy())));assert parity<1e-10;states=pd.concat(policy.states,ignore_index=True);states.to_pickle(HERE/'reports/baseline_states.pkl');write(HERE/'reports/states_manifest.json',dict(protocol_sha256=seal,states_sha256=sha(HERE/'reports/baseline_states.pkl'),rows=len(states),ledger_parity_error=parity))
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        pd.DataFrame(policy.triggers,columns=['date','ticker','rule','side','pnl','trail','age','predicted_hold_advantage']).to_csv(out/f'{arm}_triggers.csv.gz',index=False);write(done,dict(protocol_sha256=seal,ledger_sha256=sha(out/f'{arm}_ledger.csv.gz'),audit=status,trigger_count=len(policy.triggers),baseline_parity_error=parity,seconds=time.monotonic()-began));print('ACCOUNT_DONE',arm,round(time.monotonic()-began,1),flush=True)

def fit():
    from sklearn.tree import DecisionTreeRegressor,export_text
    seal=protocol();m=read(HERE/'reports/states_manifest.json');assert m['protocol_sha256']==seal and m['states_sha256']==sha(HERE/'reports/baseline_states.pkl');states=pd.read_pickle(HERE/'reports/baseline_states.pkl');labels=pd.read_pickle(V23/'data/horizon_labels.pkl')[['decision_date','ticker','label_end_h21','future_return_h21']];joined=states.merge(labels,on=['decision_date','ticker'],how='left',validate='one_to_one');valid=np.isfinite(joined.future_return_h21)&joined.label_end_h21.notna();joined=joined.loc[valid].copy();joined['advantage']=(joined.side-.0002)*joined.future_return_h21;out=HERE/'models';out.mkdir(exist_ok=True);rows=[]
    for year in range(2012,2024):
        cut=pd.Timestamp(year,1,1);train=joined.loc[joined.decision_date.ge(cut-pd.DateOffset(years=5))&joined.label_end_h21.lt(cut)&joined.decision_date.lt(cut)];assert len(train)>=10000;tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=1000,random_state=20261111);tree.fit(train[FEATURES].to_numpy(),train.advantage);f=out/f'tree_{year}.pkl';f.write_bytes(pickle.dumps(tree));detail=dict(protocol_sha256=seal,model_sha256=sha(f),state_manifest_sha256=sha(HERE/'reports/states_manifest.json'),year=year,rows=len(train),training_start=str(train.decision_date.min()),training_end=str(train.decision_date.max()),max_label_end=str(train.label_end_h21.max()),features=FEATURES,depth=tree.get_depth(),leaves=tree.get_n_leaves(),rules=export_text(tree,feature_names=FEATURES,decimals=5));write(f.with_suffix('.json'),detail);rows.append(detail);print('FIT',year,len(train),flush=True)
    write(HERE/'reports/tree_training.json',dict(models=rows,baseline_states=len(states),mature_label_states=len(joined),proxy='H21 continuation value, not globally optimal stopping'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','freeze','accounts','fit']);p.add_argument('--arms',nargs='+',default=list(ARMS));a=p.parse_args()
    if a.stage=='build':engine_build()
    elif a.stage=='freeze':print(freeze())
    elif a.stage=='fit':fit()
    else:accounts(a.arms)
