"""Annual nested, purged economic-policy fits plus matched-sample H21 Huber."""
from core import *
import pandas as pd,pickle,time,gc,argparse

def load_data():
    manifest=read(ROOT/'data_manifest.json');assert manifest['protocol_sha256']==protocol()
    for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
    meta=read(ROOT/'data/metadata.json')
    data={n:np.load(ROOT/f'data/{n}.npy',mmap_mode='r') for n in ('basis','risk','known','feasible','beta','returns','complete')}
    data.update(meta);data['dates']=pd.DatetimeIndex(meta['dates']);data['ends']=pd.DatetimeIndex(meta['label_ends'])
    return data

def choose_dates(data,cutoff,start):
    return np.flatnonzero((data['dates']>=start)&(data['dates']<cutoff)&(data['ends']<cutoff)&data['complete'])

def fit_policies(years):
    data=load_data();(ROOT/'models').mkdir(exist_ok=True);(ROOT/'account_inputs').mkdir(exist_ok=True)
    for year in years:
        started=time.monotonic();cut=pd.Timestamp(year,1,1);start=cut-pd.DateOffset(years=10);split=cut-pd.DateOffset(years=2)
        full=choose_dates(data,cut,start);inner=choose_dates(data,split,start)
        val=choose_dates(data,cut,split)
        assert len(full)>=20 and len(full)>len(inner)
        sparse=len(inner)<20 or len(val)<12
        assert data['ends'][inner].max()<split and data['ends'][full].max()<cut
        query=np.flatnonzero(data['dates'].year==year)
        for name in MODELS:
            done=ROOT/f'models/{name}_{year}.json'
            if done.exists():
                old=read(done);assert old['protocol_sha256']==protocol() and old['data_manifest_sha256']==sha(ROOT/'data_manifest.json')
                assert old['weights_sha256']==sha(ROOT/f'models/{name}_{year}.npy');continue
            k=72 if name.endswith('persist') else 36;fee=0 if name.endswith('only') else FEE
            candidates=[]
            if sparse:
                penalty=LAMBDAS[0]
            else:
                im,iq,idelta,istats=loss_inputs(data,inner,k)
                vm,vq,vdelta,vstats=loss_inputs(data,val,k)
                for penalty in LAMBDAS:
                    theta,info=solve(im,iq,idelta,penalty,fee,len(inner))
                    # Economic validation includes the real fee for all arms;
                    # the ablation removes it from fitting, not evaluation.
                    value,_=objective(theta,vm,vq,vdelta,0,FEE,len(val))
                    candidates.append(dict(penalty=penalty,validation_utility=-float(value),solver=info))
                best=max(candidates,key=lambda x:x['validation_utility']);penalty=best['penalty']
                del idelta,vdelta;gc.collect()
            mu,q,delta,stats=loss_inputs(data,full,k)
            theta,info=solve(mu,q,delta,penalty,fee,len(full));del delta;gc.collect()
            file=ROOT/f'models/{name}_{year}.npy';np.save(file,theta)
            targets=np.einsum('tnk,k->tn',data['basis'][query,:,:k],theta)
            np.save(ROOT/f'account_inputs/{name}_{year}.npy',targets)
            record=dict(model=name,year=year,protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'),
                weights_sha256=sha(file),targets_sha256=sha(ROOT/f'account_inputs/{name}_{year}.npy'),features=k,
                refit_boundary=str(cut.date()),max_training_label_end=str(data['ends'][full].max().date()),
                training_dates=[str(d.date()) for d in data['dates'][full]],inner_training_dates=len(inner),validation_dates=len(val),
                inner_max_label_end=str(data['ends'][inner].max().date()),validation_start=str(split.date()),
                query_dates=[str(d.date()) for d in data['dates'][query]],training_query_sha256=hashlib.sha256(data['dates'][full].asi8.tobytes()).hexdigest(),
                selected_penalty=penalty,selection=candidates,sparse_validation_fallback=sparse,solver=info,training_periods=len(full),cost_resets=stats,
                economic_training=True,pointwise_target_loss=False,utility_gamma=GAMMA,training_fee=fee,
                coefficients={f'{"fast" if j<36 else "persistent21"}:{data["columns"][j%36]}':float(v) for j,v in enumerate(theta)},
                future_feature_data=False,no_training_on_post2023=True)
            write(done,record)
            print('POLICY_FIT',year,name,'n',len(full),'lambda',penalty,'gap',round(info['gap'],8),flush=True)
        print('YEAR_DONE',year,round(time.monotonic()-started,1),flush=True)

def fit_aligned(years):
    import fit_horizons as original
    data=load_data();panel=pd.read_pickle(SOURCE/'data/candidate_features/panel.pkl');labels=pd.read_pickle(V23/'data/horizon_labels.pkl')
    clock=pd.DatetimeIndex(pd.read_csv(SOURCE/'data/assembled/calendar.csv',parse_dates=['Date']).Date)
    columns=data['columns'];(ROOT/'forecast_runs').mkdir(exist_ok=True)
    for year in years:
        name='sample_aligned_h21';done=ROOT/f'models/{name}_{year}.json'
        if done.exists():
            meta=read(done);assert meta['protocol_sha256']==protocol() and meta['forecast_sha256']==sha(ROOT/f'forecast_runs/{name}_{year}.csv.gz');continue
        cut=pd.Timestamp(year,1,1);selected=choose_dates(data,cut,cut-pd.DateOffset(years=10));dates=data['dates'][selected]
        train=original.common_training(panel,labels,cut,clock)
        train=train.loc[train.decision_date.isin(dates)].reset_index(drop=True)
        assert pd.DatetimeIndex(sorted(train.decision_date.unique())).equals(dates)
        # Exactly equal period influence, shared with the portfolio objective.
        train['sample_weight']=len(train)/(len(dates)*train.groupby('decision_date').ticker.transform('size'))
        query=panel.loc[original.known(panel)&panel.decision_date.dt.year.eq(year)].sort_values(['decision_date','ticker']).reset_index(drop=True)
        (x,z),pre=original.prepare(train,[query],columns,cut)
        y=original.responses(train,labels)['h21_projected_scaled'];w=train.sample_weight.to_numpy(float)
        model,scale=original.fit_huber(x,y,w);score=np.clip(model.predict(z),-8,8)
        frame=query[['decision_date','ticker']].copy();frame['score']=score;frame['refit_boundary']=cut;frame['max_training_label_end']=train.label_end_h5.max()
        file=ROOT/f'forecast_runs/{name}_{year}.csv.gz';frame.to_csv(file,index=False)
        model_file=ROOT/f'models/{name}_{year}.pkl'
        with model_file.open('wb') as f:pickle.dump(dict(model=model,preprocessing=pre,columns=columns),f,protocol=5)
        write(done,dict(model=name,year=year,protocol_sha256=protocol(),data_manifest_sha256=sha(ROOT/'data_manifest.json'),
            model_sha256=sha(model_file),forecast_sha256=sha(file),training_rows=len(train),training_periods=len(dates),
            training_dates=[str(d.date()) for d in dates],training_query_sha256=hashlib.sha256(dates.asi8.tobytes()).hexdigest(),
            refit_boundary=str(cut),max_training_label_end=str(train.label_end_h5.max()),query_rows=len(query),
            equal_period_weights=True,residual_scale=scale,pointwise_target='h21_projected_scaled'))
        print('ALIGNED_HUBER',year,len(train),len(dates),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['policy','aligned']);p.add_argument('--years',nargs='+',type=int,default=list(range(2007,2024)))
    a=p.parse_args();(fit_policies if a.action=='policy' else fit_aligned)(a.years)
