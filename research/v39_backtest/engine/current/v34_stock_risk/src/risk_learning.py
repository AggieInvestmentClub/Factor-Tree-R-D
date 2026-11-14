"""Stock-volatility and common-risk constraints: deployment versus retraining."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];V31=HERE.parent/'v31_growth_long_bias';V33=HERE.parent/'v33_adaptive_beta'
sys.path.insert(0,str(V31/'src'))
import learner as base
from settings import read,write,sha,SOURCE,V29,V23
from policy import variance
import numpy as np,pandas as pd,json,hashlib,datetime,time,gc,argparse
torch=base.torch;Policy=base.Policy
KINDS=('vol_scale','sector_neutral','corr3_neutral')
ARMS={f'{mode}_{kind}':dict(mode=mode,kind=kind) for kind in KINDS for mode in ('fixed','retrained')}
PAIRS=[[f'fixed_{k}','baseline'] for k in KINDS]+[[f'retrained_{k}',f'fixed_{k}'] for k in KINDS]+[[f'retrained_{k}','baseline'] for k in KINDS]
SECTORS=('XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','Unknown')

def protocol():
    p=HERE/'protocol.json';assert sha(p)==p.with_suffix('.sha256').read_text().strip();m=read(p)
    for n,h in m['code'].items():assert sha(HERE/'src'/n)==h,n
    for n,h in m['sources'].items():assert sha(n)==h,n
    return sha(p)

def freeze():
    p=HERE/'protocol.json'
    if p.exists():return protocol()
    files=[V31/'protocol.json',V31/'src/learner.py',V31/'src/runner.py',V29/'src/policy.py',V29/'data_manifest.json',SOURCE/'data/assembled/approval.json',SOURCE/'data/sector_candidate_snapshots.csv.gz']
    files += [V31/f'models/log_growth_{y}.pt' for y in range(2007,2024)]
    m=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='Frozen before v34 outcomes; adaptive investigation after2022 technology-long/energy-short attribution',arms=ARMS,pairs=PAIRS,
        variants={'vol_scale':'Multiply each raw score by clip(median total63-session volatility / own volatility,.5,2), before original dollar/beta projection.',
            'sector_neutral':'Project raw scores jointly to zero dollar,SPY beta,and net exposure in every source-sector group, with Unknown treated separately. Source labels are last snapshots known at the query.',
            'corr3_neutral':'Remove SPY component from63session stock returns, standardize residuals to form their correlation structure, extract top3 principal components, and project scores jointly to zero dollar,SPY beta and those3 raw-return factor exposures. Refit factors each21-session query using past data only.'},
        design='For each constraint: apply it to unchanged v31 log_growth coefficients, then refit the identical RFF model with that allocation in the objective. This separates deployment constraints from relearning.',
        unchanged='Same58price inputs+32fixed nonlinear features,21-session horizon/cadence,original efficacy gate,12% forecast-risk budget,gross3 adjusted active/500,name.15,buffer.2,nextclose,2bps/side,repaired daily engine,zero market bias.',
        fitting='100Adam epochs,.01learningrate,gradientclip1;select.1/.01ridge with repaired daily-engine log growth in prior2years;annual10year mature refit;strict inner/outer endpoint purge;fixed.1fallback2007/2008;2012claim-window shortening retained.',
        constraint_scope='Additional sector and PCA constraints are exact at scheduled desired targets. Existing daily beta/gross/eligibility guards are retained and do not impose new daily sector/PCA guards. Measure actual sector drift; do not describe the extra constraints as hard daily exposure limits.',
        inference=dict(primary='full2007-2023',secondary='2012-2023',family=9,blocks=[20,60],replications=2000,endpoint='Annual log growth difference',stress='2022 and2021-2023 descriptive, already known;no crisis-specific fitting'),
        promotion=False,limitations=['Reused history and targeted response to observed2022 failure, not independent confirmation','Same conditional universe,source sectors and valuations','Correlation estimates from63sessions are noisy;factor neutrality may remove genuine returns','Zero financing/borrow/cash yield retained;fixed-path cost sensitivities only','Gradient retains21session approximation, while selection uses actual daily accounts'],
        sources={str(f):sha(f) for f in files},code={f.name:sha(f) for f in (HERE/'src').glob('*.py') if f.name in ('risk_learning.py','test_risk_learning.py')})
    write(p,m);p.with_suffix('.sha256').write_text(sha(p));return sha(p)

def orthogonal_rows(rows,mask,width):
    n=len(mask);out=np.zeros((width,n));ix=np.flatnonzero(mask);a=np.asarray(rows)[:,ix];_,s,v=np.linalg.svd(a,full_matrices=False)
    rank=int(np.count_nonzero(s>max(a.shape)*np.finfo(float).eps*s[0]));assert rank<=width;out[:rank,ix]=v[:rank];return out,rank

def correlation_loadings(centered,market):
    f=market-market.mean();res=centered-f[:,None]*((f@centered)/float(f@f))[None,:];sigma=np.sqrt(np.sum(res*res,axis=0)/62)
    z=res/np.maximum(sigma,1e-8);gram=z@z.T/62;ev,u=np.linalg.eigh((gram+gram.T)/2);order=np.argsort(ev)[-3:][::-1];values=ev[order];assert values.min()>1e-10
    # sigma * eigenvector * sqrt(eigenvalue): raw-return loading on a unit-variance component.
    loading=(sigma[:,None]*(z.T@u[:,order])/np.sqrt(62)).T
    return loading,values/np.maximum(ev.sum(),1e-12)

def build_data():
    seal=protocol();d=base.load();out=HERE/'data';out.mkdir(exist_ok=True);dates=d['dates'];names=d['names'];t,n=len(dates),len(names)
    labels=pd.read_csv(SOURCE/'data/sector_candidate_snapshots.csv.gz',parse_dates=['date']).rename(columns={'date':'source_date','ric':'ticker'});labels=labels.loc[labels.source_date<=dates[-1]]
    grid=pd.DataFrame(dict(date=np.repeat(dates,n),ticker=np.tile(names,t),row=np.repeat(np.arange(t),n),col=np.tile(np.arange(n),t)))
    joined=pd.merge_asof(grid.sort_values('date'),labels.sort_values('source_date'),left_on='date',right_on='source_date',by='ticker',direction='backward');assert joined.source_date.isna().eq(False).sum()>0
    assert (joined.source_date.isna()|joined.source_date.le(joined.date)).all();joined['sector']=joined.sector.fillna('Unknown');codes=joined.sector.map(dict(zip(SECTORS,range(len(SECTORS)))));assert codes.notna().all()
    sector=np.empty((t,n),dtype=np.int16);sector[joined.row.to_numpy(),joined.col.to_numpy()]=codes.to_numpy()
    prices=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl');spy=prices['SPY'].loc[:dates[-1]].copy();del prices;gc.collect()
    good=spy.q.eq(1)&spy.adj_close.gt(0);same=spy.tr_segment_id.eq(spy.tr_segment_id.shift()) if 'tr_segment_id' in spy else pd.Series(True,index=spy.index)
    mr=(spy.adj_close/spy.adj_close.shift()-1).where(good&good.shift(fill_value=False)&same);mr.to_csv(out/'market_returns.csv',header=['return'])
    sectors=np.zeros((t,14,n));corr=np.zeros((t,5,n));loads=np.zeros((t,3,n));scale=np.zeros((t,n));rows=[]
    for j,date in enumerate(dates):
        mask=d['mask'][j];b=d['beta'][j];ix=np.flatnonzero(mask);ones=mask.astype(float);beta=b*mask;sector_rows=np.stack([(sector[j]==k)*mask for k in range(len(SECTORS))])
        sectors[j],rank1=orthogonal_rows(np.vstack([ones,beta,sector_rows]),mask,14)
        x=np.array(d['centered'][j])[:,ix];market=mr.loc[:date].tail(63).to_numpy();assert len(market)==63 and np.isfinite(market).all()
        loading,fractions=correlation_loadings(x,market);loads[j][:,ix]=loading;corr[j],rank2=orthogonal_rows(np.vstack([ones,beta,loads[j]]),mask,5)
        sigma=np.sqrt(d['diag'][j][ix]);scale[j,ix]=np.clip(np.median(sigma)/np.maximum(sigma,1e-8),.5,2)
        assert abs(sectors[j][:,~mask]).max(initial=0)<1e-12 and abs(corr[j][:,~mask]).max(initial=0)<1e-12
        rows.append(dict(date=str(date.date()),max_input_date=str(date.date()),window_start=str(mr.loc[:date].tail(63).index[0].date()),sector_rank=rank1,corr_rank=rank2,pc1_fraction=fractions[0],pc2_fraction=fractions[1],pc3_fraction=fractions[2],unknown_active=int((mask&(sector[j]==11)).sum())))
    for name,value in [('sector_orth',sectors),('corr_orth',corr),('factor_loadings',loads),('score_scale',scale),('sector_codes',sector)]:np.save(out/f'{name}.npy',value)
    pd.DataFrame(rows).to_csv(out/'source_audit.csv',index=False)
    write(HERE/'data_manifest.json',dict(protocol_sha256=seal,files={f.name:sha(f) for f in out.iterdir() if f.is_file()},dates=[str(x.date()) for x in dates],names=names,sectors=SECTORS,price_source_sha256=sha(SOURCE/'data/assembled/prices_candidate.pkl')))
    print('DATA_READY',t,n,flush=True)

def data(kind):
    d=base.load()
    if kind=='sector_neutral':d['orth']=np.load(HERE/'data/sector_orth.npy',mmap_mode='r')
    if kind=='corr3_neutral':d['orth']=np.load(HERE/'data/corr_orth.npy',mmap_mode='r')
    d['scale']=np.load(HERE/'data/score_scale.npy',mmap_mode='r') if kind=='vol_scale' else np.ones_like(d['mask'],dtype=float)
    return d

def tensors(d,ix):
    x=base.tensors(d,ix);x['scale']=torch.from_numpy(np.array(d['scale'][ix],copy=True));return x

def weights(model,d):
    s=model(d['x'])*d['mask']*d['scale'];u=d['orth']
    for _ in range(2):s=s-torch.bmm(u.transpose(1,2),torch.bmm(u,s.unsqueeze(-1))).squeeze(-1)
    vol=torch.sqrt(torch.clamp(variance(s,d),min=1e-20));gross=s.abs().sum(-1);peak=s.abs().amax(-1)
    factor=torch.minimum(.12/vol,torch.minimum(3*torch.clamp(d['mask'].sum(-1)/500,max=1)/gross.clamp(min=1e-15),.15/peak.clamp(min=1e-15)))
    return s*(factor*d['gate']*(1-1e-12)).unsqueeze(-1)

def predict(model,d,ix):
    with torch.no_grad():return weights(model,tensors(d,ix)).numpy()

def train(d,ix,penalty):
    assert d['complete'][ix].all() and np.isfinite(d['returns'][ix][d['mask'][ix]]).all();x=tensors(d,ix);m=Policy('replay_random');opt=torch.optim.Adam(m.parameters(),lr=.01);trace=[]
    for epoch in range(100):
        opt.zero_grad();loss=base.loss(weights(m,x),x,'log_growth',m,penalty);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if epoch%20==0 or epoch==99:trace.append(dict(epoch=epoch,loss=float(loss.detach())))
    with torch.no_grad():stats=base.loss(weights(m,x),x,'log_growth',details=True)
    stats['trace']=trace;return m,stats

def fit(years):
    seal=protocol();manifest=sha(HERE/'data_manifest.json');datasets={k:data(k) for k in KINDS};loaded=None
    for year in years:
        cut=pd.Timestamp(year,1,1);start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2)
        for kind,d in datasets.items():
            done=HERE/f'models/{kind}_{year}.json'
            if done.exists():assert read(done)['protocol_sha256']==seal;continue
            began=time.monotonic();full=base.selected(d,cut,start);inner=base.selected(d,split,start);val=np.flatnonzero((d['dates']>=split)&(d['dates']<cut));query=np.flatnonzero(d['dates'].year==year);fallback=year<2009 or len(inner)<20;choices=[]
            if not fallback:
                if loaded is None:loaded=base.inputs()
                for penalty in (.1,.01):
                    m,stats=train(d,inner,penalty);target=pd.DataFrame(predict(m,d,val),index=d['dates'][val],columns=d['names']);v=base.validation(loaded,target,loaded['clock'][loaded['clock']<cut][-1],'log_growth');choices.append(dict(penalty=penalty,validation=v,training=stats));print('INNER',year,kind,penalty,round(v['utility'],5),flush=True)
                penalty=max(choices,key=lambda c:c['validation']['utility'])['penalty']
            else:penalty=.1
            m,stats=train(d,full,penalty);file=HERE/f'models/{kind}_{year}.pt';file.parent.mkdir(exist_ok=True);torch.save(m.state_dict(),file);f=HERE/f'account_inputs/{kind}_{year}.npy';f.parent.mkdir(exist_ok=True);np.save(f,predict(m,d,query))
            write(done,dict(kind=kind,year=year,protocol_sha256=seal,data_manifest_sha256=manifest,weights_sha256=sha(file),targets_sha256=sha(f),training_periods=len(full),training_query_sha256=hashlib.sha256(d['dates'][full].asi8.tobytes()).hexdigest(),max_training_label_end=str(d['ends'][full].max()),inner_max_training_label_end=str(d['ends'][inner].max()),validation_start=str(split),fallback=fallback,selected_penalty=penalty,selection=choices,training=stats,seconds=time.monotonic()-began))
            print('FIT_DONE',year,kind,round(time.monotonic()-began,1),flush=True);gc.collect()

def prepare(arms):
    seal=protocol();out=HERE/'account_inputs';out.mkdir(exist_ok=True)
    for arm in arms:
        a=ARMS[arm];d=data(a['kind']);chunks=[]
        for year in range(2007,2024):
            if a['mode']=='retrained':
                m=read(HERE/f'models/{a["kind"]}_{year}.json');f=out/f'{a["kind"]}_{year}.npy';assert m['protocol_sha256']==seal and m['targets_sha256']==sha(f);chunks.append(np.load(f))
            else:
                file=V31/f'models/log_growth_{year}.pt';m=Policy('replay_random');m.load_state_dict(torch.load(file,weights_only=True));chunks.append(predict(m,d,np.flatnonzero(d['dates'].year==year)))
        ix=np.flatnonzero(d['dates']>=pd.Timestamp(d['anchor']));w=np.concatenate(chunks);assert abs(np.einsum('tkn,tn->tk',d['orth'][ix],w)).max()<1e-9 and abs(w[~d['mask'][ix]]).max(initial=0)<1e-10
        f=out/f'{arm}_desired.pkl';pd.DataFrame(w,index=d['dates'][ix],columns=d['names']).to_pickle(f);write(out/f'{arm}.json',dict(protocol_sha256=seal,target_sha256=sha(f),settings=a));print('PREPARED',arm,flush=True)

def accounts(arms):
    seal=protocol();loaded=base.inputs();out=HERE/'accounts';out.mkdir(exist_ok=True);d=base.load();sector=np.load(HERE/'data/sector_codes.npy');ticker_index={n:j for j,n in enumerate(d['names'])}
    from runner import bridge,simulate
    for arm in arms:
        done=out/f'{arm}_complete.json'
        if done.exists():assert read(done)['ledger_sha256']==sha(out/f'{arm}_ledger.csv.gz');continue
        f=HERE/f'account_inputs/{arm}_desired.pkl';m=read(f.with_name(f'{arm}.json'));assert sha(f)==m['target_sha256'] and m['protocol_sha256']==seal
        noncash=bridge.NoncashExposureAudit(pd.read_csv(SOURCE/'data/distribution_audit/spin_census_dispositions.csv'));count=0;exposure=[]
        def sink(frame):
            nonlocal count
            noncash.consume(frame);count+=len(frame);ix=d['dates'].searchsorted(pd.DatetimeIndex(frame.Date),side='right')-1;ns=frame.ticker.map(ticker_index).to_numpy();x=frame[['Date','closing_value']].copy();x['sector']=sector[ix,ns];exposure.append(x.groupby(['Date','sector']).closing_value.sum())
        began=time.monotonic();print('ACCOUNT_BEGIN',arm,flush=True);ledger,pos,actual,decisions,status=simulate(loaded,pd.read_pickle(f),repair=True,positions=True,sink=sink)
        assert pos.empty and status['streamed_position_rows']==count and status['arithmetic_passed'];status['noncash_exposure_audit']=noncash.report()
        if noncash.rows:status['issues'].extend(noncash.rows);status['validated_performance']=False
        exp=pd.concat(exposure).groupby(level=[0,1]).sum().unstack('sector').reindex(columns=range(12),fill_value=0).divide(ledger.nav,axis=0);exp.columns=SECTORS;exp.to_csv(out/f'{arm}_sector_exposure.csv.gz')
        for suffix,frame,index in [('ledger',ledger,True),('targets',actual,True),('decisions',decisions,False)]:frame.to_csv(out/f'{arm}_{suffix}.csv.gz',index=index)
        write(done,dict(protocol_sha256=seal,arm=arm,settings=ARMS[arm],target_sha256=sha(f),ledger_sha256=sha(out/f'{arm}_ledger.csv.gz'),audit=status,seconds=time.monotonic()-began));print('ACCOUNT_DONE',arm,round(time.monotonic()-began,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','data','fit','prepare','accounts']);p.add_argument('--years',type=int,nargs='+',default=list(range(2007,2024)));p.add_argument('--arms',nargs='+',default=list(ARMS));a=p.parse_args()
    if a.stage=='freeze':print(freeze())
    elif a.stage=='data':build_data()
    elif a.stage=='fit':fit(a.years)
    elif a.stage=='prepare':prepare(a.arms)
    else:accounts(a.arms)
