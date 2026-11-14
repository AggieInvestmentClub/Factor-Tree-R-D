"""Causal characteristic histories and exact source-risk bases. Labels separate."""
from core import *
import pandas as pd,time,gc
from lazy_covariances import build as build_covariances
from model_core_long import known

def main():
    seal=protocol();spec=read(ROOT/'protocol.json');out=ROOT/'data';out.mkdir(exist_ok=True)
    panel_path=SOURCE/'data/candidate_features/panel.pkl';label_path=V23/'data/horizon_labels.pkl'
    assert sha(panel_path)==spec['sources']['panel'] and sha(label_path)==spec['sources']['labels']
    panel=pd.read_pickle(panel_path);labels=pd.read_pickle(label_path)
    assert panel.index.equals(labels.index) and panel[['decision_date','ticker']].equals(labels[['decision_date','ticker']])
    clock=pd.DatetimeIndex(pd.read_csv(SOURCE/'data/assembled/calendar.csv',parse_dates=['Date']).Date)
    names=sorted(panel.loc[known(panel),'ticker'].unique());columns=spec['features'];k=len(columns)
    anchor=pd.Timestamp(read(V23/'account_inputs/covariances_cadence21/manifest.json')['dates'][0])
    dates=pd.DatetimeIndex(sorted(panel.loc[known(panel),'decision_date'].unique()))
    schedule=dates[(clock.get_indexer(dates)-clock.get_loc(anchor))%21==0]
    original_dates=pd.DatetimeIndex(read(V23/'account_inputs/covariances_cadence21/manifest.json')['dates'])
    assert schedule[schedule>=anchor].equals(original_dates)
    fields=[f'{c}_h{h}' for h in (5,10,21) for c in ('future_return','future_parent_return','future_spy','target_risk_resid')]
    valid=np.isfinite(labels[fields].to_numpy(float)).all(axis=1)
    valid &= panel.v8_target_absolute_h5.notna().to_numpy() & panel.v8_target_spy_h5.notna().to_numpy()
    quality=pd.DataFrame(dict(date=panel.decision_date,known=known(panel),valid=valid))
    complete=quality.loc[quality.known].groupby('date').valid.all().reindex(schedule).to_numpy(bool,copy=True)
    counts=quality.loc[quality.known].groupby('date').size().reindex(schedule).to_numpy();complete &= counts>=10
    ends=labels.groupby('decision_date').label_end_h21.max().reindex(schedule)
    future=labels.assign(known=known(panel).to_numpy()).loc[lambda f:f.known & f.decision_date.isin(schedule)]
    returns=future.pivot(index='decision_date',columns='ticker',values='future_return_h21').reindex(index=schedule,columns=names).to_numpy(float)
    labels_index=panel.index.to_numpy();del labels,future,quality;gc.collect()
    print('BUILD_COVARIANCES',len(schedule),flush=True)
    prices=pd.read_pickle(SOURCE/'data/assembled/prices_candidate.pkl')
    approval=read(SOURCE/'data/assembled/approval.json')
    cov=build_covariances(prices,names,schedule,out/'covariances',approval['prices_sha256'])
    del prices;gc.collect()
    basis=np.lib.format.open_memmap(out/'basis.npy',mode='w+',dtype='float64',shape=(len(schedule),len(names),2*k))
    risk=np.zeros((len(schedule),2*k,2*k));masks=np.zeros((len(schedule),len(names)),bool)
    betas=np.full((len(schedule),len(names)),np.nan);feasible=np.zeros_like(masks)
    state=np.full((2,len(names),k),np.nan);alphas=(.25,spec['policy']['persistence_alpha'])
    ticker_map={v:i for i,v in enumerate(names)};date_map={v:i for i,v in enumerate(schedule)}
    risk_parity=[];started=time.monotonic()
    # Only decision-known fields enter the causal feature loop.
    source=panel.loc[known(panel),['decision_date','ticker','beta_spy63',*columns]].copy()
    del panel;gc.collect()
    for number,(date,group) in enumerate(source.groupby('decision_date',sort=True)):
        ix=group.ticker.map(ticker_map).to_numpy(int);mask=np.zeros(len(names),bool);mask[ix]=True
        raw=group[columns].to_numpy(float);raw=np.nan_to_num(raw,nan=0,posinf=0,neginf=0)
        update_states(state,raw,ix,mask,alphas)
        if date not in date_map:continue
        j=date_map[date];beta=np.full(len(names),np.nan);beta[ix]=group.beta_spy63.to_numpy(float)
        path=out/f'covariances/{date:%Y-%m-%d}.npz'
        with np.load(path,allow_pickle=False) as z:indices=z['indices'];centered=z['centered'];meta=json.loads(str(z['metadata']))
        assert pd.Timestamp(meta['max_source_date'])<=date
        values=np.concatenate([state[0],state[1]],axis=1)
        b,q,active=neutral_basis(values,beta,mask,centered,indices)
        basis[j]=b;risk[j]=q;masks[j]=mask;betas[j]=beta;feasible[j]=active
        assert abs(b).sum(axis=0).max()<=3*min(1,active.sum()/500)+1e-9
        assert abs(b).max()<=.15+1e-9 and np.diag(q).max()<=.12**2+1e-9
        if date in original_dates[[0,len(original_dates)//2,-1]]:
            from risk_engine import allocate_target
            record=cov[date]
            for c in (0,k-1,k):
                old,_=allocate_target(pd.Series(np.nan_to_num(values[:,c]),index=names),pd.Series(beta,index=names),record,
                    pd.Series(mask,index=names),date=date,vol_target=.12,gross_cap=3,name_cap=.15)
                error=float(np.max(abs(old.to_numpy()-b[:,c])));assert error<1e-10
                risk_parity.append(dict(date=str(date.date()),column=c,max_error=error))
        if j%50==0:print('BASIS',j,len(schedule),round(time.monotonic()-started,1),flush=True)
    basis.flush()
    for name,value in [('risk',risk),('known',masks),('feasible',feasible),('beta',betas),('returns',returns),('complete',complete)]:np.save(out/f'{name}.npy',value)
    write(out/'metadata.json',dict(dates=[str(d.date()) for d in schedule],label_ends=[None if pd.isna(d) else str(d.date()) for d in ends],
        names=names,columns=columns,anchor=str(anchor.date()),features=2*k,feature_banks=['EWMA.25','EWMA_half_life21'],
        price_only=True,no_future_fields_in_feature_loop=True,risk_parity=risk_parity,
        complete_queries=int(complete.sum()),scheduled_queries=len(schedule),prediction_queries_filtered_by_labels=False))
    files=list(out.glob('*.npy'))+[out/'metadata.json',out/'covariances/manifest.json']
    write(ROOT/'data_manifest.json',dict(protocol_sha256=seal,files={str(f.relative_to(ROOT)):sha(f) for f in files},
        source_price_hash=approval['prices_sha256'],source_panel_hash=spec['sources']['panel'],source_label_hash=spec['sources']['labels']))
    print('DATA_DONE',len(schedule),int(complete.sum()),flush=True)

if __name__=='__main__':main()
