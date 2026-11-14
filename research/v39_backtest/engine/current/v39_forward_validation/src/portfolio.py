"""Frozen annual refits and two forward accounts; no search over exit settings."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('v39_config',Path(__file__).with_name('common.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
c.sys.path.insert(0,str(c.V31/'src'));c.sys.path.insert(0,str(c.V34/'src'));c.sys.path.insert(0,str(c.V38/'src'))
import learner,exits,risk_learning
from allocation import target,annual_variance,unit_market
from settings import selected
from policy import Policy,torch
import numpy as np,pandas as pd,gc,time,argparse,hashlib

def data():
    out=c.HERE/'data/model';m=c.read(out/'metadata.json');d={k:np.load(out/f'{k}.npy',mmap_mode='r') for k in ('x','centered','diag','orth','mask','beta','returns','complete','gate','corr_orth','factor_loadings')};d.update(m);d['dates']=pd.DatetimeIndex(m['dates']);d['ends']=pd.DatetimeIndex(m['label_ends']);return d

def inputs():
    folder=c.HERE/'data/assembled';approval=c.read(folder/'approval.json');assert approval['source_qualification_passed'] and approval['prices_sha256']==c.sha(folder/'prices_candidate.pkl') and approval['events_sha256']==c.sha(folder/'events.csv')
    m=c.read(c.HERE/'data/model/metadata.json');prices=pd.read_pickle(folder/'prices_candidate.pkl');normalized=0
    for name,f in prices.items():
        missing=f.distribution_reinvestment_turnover.isna()
        # Calendar extension creates empty action metadata outside an issue's
        # observed history. No distribution trade exists on those empty rows.
        # Quotes/returns/eligibility stay missing and retain all exposure audits.
        assert f.loc[missing,'q'].ne(1).all();normalized+=int(missing.sum());f.loc[missing,'distribution_reinvestment_turnover']=0.
    c.write(c.HERE/'data/account_input_normalization.json',dict(action_metadata_empty_rows=normalized,price_or_return_values_filled=False,policy='Zero distribution-trade turnover only on rows without qualified quotes; no return or wealth imputation',source_price_sha256=approval['prices_sha256']))
    return dict(prices=prices,events=pd.read_csv(folder/'events.csv',parse_dates=['effective_date']),clock=pd.DatetimeIndex(m['calendar']),names=m['names'],beta=pd.read_pickle(c.HERE/'data/model/daily_beta.pkl'),eligible=pd.read_pickle(c.HERE/'data/model/daily_eligible.pkl'),claims=c.read(folder/'claims.json'))

def simulate(loaded,targets,end,bias=False,start=None,sink=None,policy=None,cutoff=c.BOUNDARY):
    clock=loaded['clock'];start=clock[clock.get_loc(targets.index[0])+1] if start is None else pd.Timestamp(start);claims=[r for r in loaded['claims'] if start<=pd.Timestamp(r['ex_date'])<=end];assert all(pd.Timestamp(r['settlement_date'])<=end for r in claims)
    return exits.engine().simulate_stock_portfolio(loaded['prices'],targets,loaded['events'],start=start,end=end,cost_bps=2,buffer_l1=.2,risk_betas=loaded['beta'],eligibility=loaded['eligible'],neutral=not bias,dollar_neutral=not bias,beta_neutral=not bias,controlled_long_bias=bias,repair_stale_risk=True,position_cap=.15,gross_cap=3,gross_guard=3.3,data_cutoff=cutoff,record_positions=sink is not None,position_sink=sink,position_batch_rows=1,distribution_claims=claims,position_exit_policy=policy)

def validation(loaded,targets,end,cut):
    crossed=[pd.Timestamp(r['ex_date']) for r in loaded['claims'] if targets.index[0]<pd.Timestamp(r['ex_date'])<=end<pd.Timestamp(r['settlement_date'])]
    if crossed:end=loaded['clock'][loaded['clock']<min(crossed)][-1];targets=targets.loc[targets.index<end]
    ledger,_,_,_,audit=simulate(loaded,targets,end,cutoff=str(cut.date()));assert audit['arithmetic_passed'];r=ledger.daily_return.to_numpy();return dict(utility=float(252*np.log1p(r).mean()),annual_mean=float(252*r.mean()),annual_log_growth=float(252*np.log1p(r).mean()),annual_variance=float(252*r.var(ddof=1)),start=str(ledger.index[0]),end=str(ledger.index[-1]),sessions=len(r),claim_boundary_shortened=bool(crossed),exact_repaired_engine=True,validated_performance=audit['validated_performance'],issues=audit['issues'])

def fit():
    d=data();loaded=None
    for year in (2025,2026):
        done=c.HERE/f'models/log_growth_{year}.json'
        if done.exists():assert c.read(done)['design_sha256']==c.sha(c.HERE/'design.json');continue
        began=time.monotonic();cut=pd.Timestamp(year,1,1);start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2);full=selected(d,cut,start);inner=selected(d,split,start);val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut));choices=[]
        if loaded is None:loaded=inputs()
        for penalty in (.1,.01):
            model,stats=learner.train(d,inner,'log_growth',penalty);targets=pd.DataFrame(learner.predict(model,d,val),index=d['dates'][val],columns=d['names']);v=validation(loaded,targets,loaded['clock'][loaded['clock']<cut][-1],cut);choices.append(dict(penalty=penalty,validation=v,training=stats));print('INNER',year,penalty,flush=True)
        penalty=max(choices,key=lambda x:x['validation']['utility'])['penalty'];model,stats=learner.train(d,full,'log_growth',penalty);file=done.with_suffix('.pt');torch.save(model.state_dict(),file)
        c.write(done,dict(year=year,objective='log_growth',design_sha256=c.sha(c.HERE/'design.json'),weights_sha256=c.sha(file),training_periods=len(full),training_query_sha256=hashlib.sha256(d['dates'][full].asi8.tobytes()).hexdigest(),max_training_label_end=str(d['ends'][full].max()),refit_boundary=str(cut),inner_max_training_label_end=str(d['ends'][inner].max()),validation_start=str(split),selection=choices,selected_penalty=penalty,fallback=False,training=stats,seconds=time.monotonic()-began,array_qa_sha256=c.sha(c.HERE/'data/array_qa.json')));print('FIT_DONE',year,flush=True)

def prepare():
    d=data();corr=dict(d);corr['orth']=d['corr_orth'];corr['scale']=np.ones_like(d['mask'],dtype=float);frames=[];records=[]
    for year in (2024,2025,2026):
        file=c.HERE/f'models/log_growth_{year}.pt';m=c.read(file.with_suffix('.json'));assert m['weights_sha256']==c.sha(file);model=Policy('replay_random');model.load_state_dict(torch.load(file,weights_only=True));ix=np.flatnonzero(d['dates'].year==year);neutral=risk_learning.predict(model,corr,ix);values=[]
        for row,j in enumerate(ix):
            gate=d['gate'][j];mask=d['mask'][j];beta=d['beta'][j];indices=np.flatnonzero(np.any(d['centered'][j]!=0,axis=0));centered=d['centered'][j][:,indices];a=neutral[row]/gate if gate else np.zeros(len(mask));w,info=target(a,beta,mask,centered,indices,.12,.5,gate);alpha=w-unit_market(beta,mask)*info['effective_bias'];assert abs(d['factor_loadings'][j]@alpha).max()<1e-9
            assert annual_variance(w,centered,indices)<=.12**2+1e-9;values.append(w);records.append(dict(date=str(d['dates'][j].date()),model_year=year,gate=gate,gross=float(abs(w).sum()),**info))
        frames.append(pd.DataFrame(values,index=d['dates'][ix],columns=d['names']))
    # Start the new account in cash onJan2. Carry the latest scheduled source target
    # to the immediately prior decision close so its first order is executableJan2.
    old=pd.read_pickle(c.V36/'account_inputs/fixed_corr_long50_desired.pkl');initial=old.iloc[[-1]].reindex(columns=d['names'],fill_value=0);origin=initial.index[0];initial.index=pd.DatetimeIndex(['2023-12-29']);desired=pd.concat([initial]+frames);out=c.HERE/'account_inputs';out.mkdir(exist_ok=True);desired.to_pickle(out/'desired.pkl');pd.DataFrame(records).to_csv(out/'target_diagnostics.csv',index=False)
    c.write(out/'desired.json',dict(design_sha256=c.sha(c.HERE/'design.json'),target_sha256=c.sha(out/'desired.pkl'),first_execution=c.START,initial_source_forecast=str(origin.date()),initial_decision='2023-12-29',forward_forecasts=len(desired)-1,original_cadence_retained=True));print('TARGETS_READY',len(desired),flush=True)

class ExitPolicy(exits.Policy):
    def __init__(self,loaded,desired):
        self.rule='vol2sigma';self.names=list(desired.columns);self.desired=desired;self.loaded=loaded;self.collect=False;m=c.read(c.HERE/'data/model/metadata.json');assert m['names']==self.names;self.clock=pd.DatetimeIndex(m['calendar']);self.returns=np.load(c.HERE/'data/model/qualified_returns.npy',mmap_mode='r');self.sdates=pd.DatetimeIndex(m['dates']);self.loads=np.load(c.HERE/'data/model/factor_loadings.npy',mmap_mode='r');n=len(self.names)
        self.entry=np.full(n,np.nan);self.peak=np.full(n,np.nan);self.shares=np.zeros(n);self.side=np.zeros(n);self.age=np.zeros(n);self.entrysigma=np.zeros(n);self.blocked=np.zeros(n,bool);self.current=None;self.valid=np.zeros(n,bool);self.states=[];self.triggers=[];self.models={};self.last_new=None;self.name_index={n:i for i,n in enumerate(self.names)};self.seal=c.sha(c.HERE/'design.json')
    def __call__(self,*args,**kwargs):
        before=len(self.triggers);result=super().__call__(*args,**kwargs)
        # Audit enrichment only: inherited trigger, sizing and re-entry are unchanged.
        for record in self.triggers[before:]:
            j=self.name_index[record['ticker']];record.update(entry_sigma21=float(self.entrysigma[j]),threshold=float(np.clip(2*self.entrysigma[j],.03,.20)))
        return result

def accounts():
    loaded=inputs();desired=pd.read_pickle(c.HERE/'account_inputs/desired.pkl');out=c.HERE/'accounts';out.mkdir(exist_ok=True)
    for arm in ('benchmark','vol2sigma'):
        done=out/f'{arm}_complete.json'
        if done.exists():assert c.read(done)['ledger_sha256']==c.sha(out/f'{arm}_ledger.csv.gz');continue
        policy=ExitPolicy(loaded,desired) if arm=='vol2sigma' else None;counts=0;event_positions=[]
        census=pd.read_csv(c.HERE/'data/assembled/noncash_census.csv',parse_dates=['date'])
        def sink(frame):
            nonlocal counts
            counts+=len(frame)
            if policy:policy.consume(frame)
            date=pd.Timestamp(frame.Date.iloc[0]);relevant=census[census.date.eq(date)]
            if len(relevant):event_positions.append(frame[frame.ticker.isin(relevant.ticker)])
            for ticker,boundary in [('CMA.N^B26','2025-01-01'),('QRVO.OQ^J26','2025-01-01')]:
                if date>=pd.Timestamp(boundary):
                    z=frame[frame.ticker.eq(ticker)];assert not z.previous_closing_shares.ne(0).any() and not z.closing_shares.ne(0).any(),('Unmodeled later terminal action has account exposure',ticker,date)
        began=time.monotonic();print('ACCOUNT_BEGIN',arm,flush=True);ledger,pos,actual,decisions,audit=simulate(loaded,desired,pd.Timestamp(c.END),bias=True,start=c.START,sink=sink,policy=policy)
        assert pos.empty and audit['streamed_position_rows']==counts and audit['arithmetic_passed']
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        if event_positions:pd.concat(event_positions).to_csv(out/f'{arm}_noncash_positions.csv.gz',index=False)
        triggers=policy.triggers if policy else [];pd.DataFrame(triggers,columns=['date','ticker','rule','side','pnl','trail','age','predicted_hold_advantage','entry_sigma21','threshold']).to_csv(out/f'{arm}_triggers.csv.gz',index=False)
        c.write(done,dict(arm=arm,design_sha256=c.sha(c.HERE/'design.json'),ledger_sha256=c.sha(out/f'{arm}_ledger.csv.gz'),desired_sha256=c.sha(c.HERE/'account_inputs/desired.pkl'),audit=audit,trigger_count=len(triggers),seconds=time.monotonic()-began));print('ACCOUNT_DONE',arm,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','prepare','accounts']);a=p.parse_args();globals()[a.stage]()
